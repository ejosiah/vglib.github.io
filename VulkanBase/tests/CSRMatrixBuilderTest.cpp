#include "VulkanFixture.hpp"
#include "linalg/factory.hpp"
#include "linalg/gpu/csr_matrix.hpp"
#include "linalg/matrix_market.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

namespace {
    static_assert(sizeof(gpu::linalg::SourceEntry) == 12);

    constexpr uint32_t WorkGroupSize = 32;
    constexpr uint32_t randomRows = 64;
    constexpr uint32_t randomCols = 64;

    std::vector<gpu::linalg::SourceEntry> createSourceEntries(const matrix_market::matrix& source) {
        return map_range(source, [](const auto& entry) {
            return gpu::linalg::SourceEntry{
                .value = entry.value,
                .row = entry.rowIndex,
                .col = entry.colIndex,
            };
        });
    }

    std::vector<int32_t> createFlags(const matrix_market::matrix& source) {
        std::vector<int32_t> flags(source.rowCount * source.colCount, 0);

        for(const auto [row, col, value] : source) {
            flags[row * source.colCount + col] = 1;
        }

        return flags;
    }

    matrix_market::matrix createRandomSource(uint32_t entryCount, uint32_t seed) {
        std::mt19937 rng{seed};
        std::uniform_real_distribution<float> values{-10.0f, 10.0f};
        std::vector<uint32_t> indices(randomRows * randomCols);
        std::iota(indices.begin(), indices.end(), 0u);
        std::shuffle(indices.begin(), indices.end(), rng);

        auto entries = map_range(indices.begin(), indices.begin() + entryCount, [&](uint32_t index) {
            return matrix_market::entry{
                .rowIndex = index / randomCols,
                .colIndex = index % randomCols,
                .value = values(rng),
            };
        });

        return {
            .data = std::move(entries),
            .rowCount = randomRows,
            .colCount = randomCols,
        };
    }
}

class CSRMatrixBuilderTest : public VulkanFixture {};

template<typename CpuMatrix>
void expectMatchesCpuCsrBuilder(VulkanDevice& device, auto&& execute, const matrix_market::matrix& source) {
    const auto expected = linalg::create_matrix<CpuMatrix>(source);
    const auto sourceEntries = createSourceEntries(source);
    const auto flags = createFlags(source);

    const std::vector<float> values(expected.data.size(), 0.0f);
    const std::vector<uint32_t> colIndices(expected.col_indices.size(), 0);
    const std::vector<uint32_t> rowOffsets(source.rowCount + 1, 0);
    const std::vector<uint32_t> counts{source.rowCount, source.colCount, 0};

    constexpr auto outputUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    constexpr auto sourceUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    constexpr auto flagsUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    auto valuesBuffer = device.createCpuVisibleBuffer(values.data(), BYTE_SIZE(values), outputUsage);
    auto colIndicesBuffer = device.createCpuVisibleBuffer(colIndices.data(), BYTE_SIZE(colIndices), outputUsage);
    auto rowOffsetsBuffer = device.createCpuVisibleBuffer(rowOffsets.data(), BYTE_SIZE(rowOffsets), outputUsage);
    auto countsBuffer = device.createCpuVisibleBuffer(counts.data(), BYTE_SIZE(counts), outputUsage);
    const auto sourceCount = static_cast<uint32_t>(sourceEntries.size());
    auto sourceBuffer = device.createBuffer(sourceUsage, VMA_MEMORY_USAGE_CPU_TO_GPU, sizeof(sourceCount) + BYTE_SIZE(sourceEntries));
    sourceBuffer.copy(&sourceCount, sizeof(sourceCount));
    sourceBuffer.copy(sourceEntries.data(), BYTE_SIZE(sourceEntries), sizeof(sourceCount));
    auto flagsBuffer = device.createCpuVisibleBuffer(flags.data(), BYTE_SIZE(flags), flagsUsage);

    gpu::linalg::CSRMatrix gpuMatrix{
        .values = valuesBuffer,
        .colIndices = colIndicesBuffer,
        .rowOffsets = rowOffsetsBuffer,
        .counts = countsBuffer,
        .numRows = source.rowCount,
        .numCols = source.colCount,
    };
    gpu::linalg::CSRMatrixBuilder builder{device};
    builder.init(gpuMatrix, sourceBuffer, flagsBuffer);

    execute([&](auto commandBuffer) {
        builder.build(commandBuffer, gpuMatrix, sourceBuffer, flagsBuffer);
    });

    const auto actualValues = valuesBuffer.span<float>(expected.data.size());
    const auto actualCols = colIndicesBuffer.span<uint32_t>(expected.col_indices.size());
    const auto actualOffsets = rowOffsetsBuffer.span<uint32_t>(source.rowCount + 1);
    const auto actualCounts = countsBuffer.span<uint32_t>(3);

    ASSERT_EQ(source.rowCount, actualCounts[0]);
    ASSERT_EQ(source.colCount, actualCounts[1]);
    ASSERT_EQ(expected.data.size(), actualCounts[2]);
    for(size_t i = 0; i < expected.row_offsets.size(); ++i) {
        ASSERT_EQ(expected.row_offsets[i], actualOffsets[i]);
    }

    for(uint32_t row = 0; row < source.rowCount; ++row) {
        const auto expectedStart = expected.row_offsets[row];
        const auto expectedEnd = expected.row_offsets[row + 1];
        const auto actualStart = actualOffsets[row];
        const auto actualEnd = actualOffsets[row + 1];

        ASSERT_EQ(expectedEnd - expectedStart, actualEnd - actualStart);
        for(auto i = expectedStart; i < expectedEnd; ++i) {
            const auto expectedCol = expected.col_indices[i];
            const auto found = std::find(actualCols.begin() + actualStart, actualCols.begin() + actualEnd, expectedCol);
            ASSERT_NE(actualCols.begin() + actualEnd, found) << "missing column " << expectedCol << " in row " << row;

            const auto actualIndex = static_cast<size_t>(std::distance(actualCols.begin(), found));
            ASSERT_FLOAT_EQ(expected.data[i], actualValues[actualIndex]) << "row " << row << " col " << expectedCol;
        }
    }

    countsBuffer.unmap();
    rowOffsetsBuffer.unmap();
    colIndicesBuffer.unmap();
    valuesBuffer.unmap();
}

TEST_F(CSRMatrixBuilderTest, matchesCpuCsrBuilder) {
    constexpr uint32_t rows = 5;
    constexpr uint32_t cols = 6;
    using CpuMatrix = linalg::sparse_csr_matrix<float, rows, cols>;

    const matrix_market::matrix source{
        .data = {
            {4, 5, 4.5f},
            {0, 2, 1.25f},
            {2, 1, -3.0f},
            {0, 0, 5.0f},
            {3, 4, 8.0f},
            {2, 5, 7.0f},
            {4, 1, 2.0f},
        },
        .rowCount = rows,
        .colCount = cols,
    };

    expectMatchesCpuCsrBuilder<CpuMatrix>(device, [this](auto&& func) { execute(std::forward<decltype(func)>(func)); }, source);
}

TEST_F(CSRMatrixBuilderTest, matchesCpuCsrBuilderWithManyFullWorkGroups) {
    using CpuMatrix = linalg::sparse_csr_matrix<float, randomRows, randomCols>;

    const auto source = createRandomSource(WorkGroupSize * 20, 0xC5A5u);

    expectMatchesCpuCsrBuilder<CpuMatrix>(device, [this](auto&& func) { execute(std::forward<decltype(func)>(func)); }, source);
}

TEST_F(CSRMatrixBuilderTest, matchesCpuCsrBuilderWithPartialWorkGroup) {
    using CpuMatrix = linalg::sparse_csr_matrix<float, randomRows, randomCols>;

    const auto source = createRandomSource(WorkGroupSize * 20 + 15, 0xC5A6u);

    expectMatchesCpuCsrBuilder<CpuMatrix>(device, [this](auto&& func) { execute(std::forward<decltype(func)>(func)); }, source);
}
