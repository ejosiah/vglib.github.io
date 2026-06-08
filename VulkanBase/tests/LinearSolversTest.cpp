#include "VulkanFixture.hpp"
#include "gpu/algorithm.h"
#include "linalg/gpu/solver.hpp"
#include "linalg/linear_system.hpp"
#include "linalg/solver.hpp"

#include <filesystem>
#include <vector>

namespace {
    constexpr uint32_t systemSize = 256;
    constexpr uint32_t solverIterations = 30;

    using CpuMatrix = linalg::sparse_csr_matrix<float, systemSize, systemSize>;
    using CpuVector = linalg::vector<float, systemSize>;

    std::filesystem::path dataPath(const std::filesystem::path& relativePath) {
        return std::filesystem::path{__FILE__}.parent_path().parent_path().parent_path() / "data" / relativePath;
    }

    CpuVector redBlackGaussSeidel(const CpuMatrix& A, const CpuVector& b, uint32_t numIterations) {
        auto x = linalg::create_matrix<CpuVector>();

        for(uint32_t iteration = 0; iteration < numIterations; ++iteration) {
            for(uint32_t pass = 0; pass < 2; ++pass) {
                for(uint32_t row = 0; row < systemSize; ++row) {
                    if((row & 1u) != pass) {
                        continue;
                    }

                    float diagonal{};
                    float sum{};
                    const auto start = A.row_offsets[row];
                    const auto end = A.row_offsets[row + 1];

                    for(auto i = start; i < end; ++i) {
                        const auto col = A.col_indices[i];
                        const auto value = A.data[i];

                        if(col == row) {
                            diagonal = value;
                        } else {
                            sum += value * x(col);
                        }
                    }

                    if(diagonal != 0.0f) {
                        x(row) = (b(row) - sum) / diagonal;
                    }
                }
            }
        }

        return x;
    }
}

class LinearSolversTest : public VulkanFixture {
protected:
    void postVulkanInit() override {
        gpu::init(device, _fileManager);
    }

    void TearDown() override {
        gpu::shutdown();
    }

    struct GpuSystem {
        VulkanBuffer values;
        VulkanBuffer colIndices;
        VulkanBuffer rowOffsets;
        VulkanBuffer rhs;
    };

    GpuSystem createGpuSystem(const CpuMatrix& A, const CpuVector& b) {
        constexpr auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        return {
            device.createCpuVisibleBuffer(A.data.data(), BYTE_SIZE(A.data), usage),
            device.createCpuVisibleBuffer(A.col_indices.data(), BYTE_SIZE(A.col_indices), usage),
            device.createCpuVisibleBuffer(A.row_offsets.data(), BYTE_SIZE(A.row_offsets), usage),
            device.createCpuVisibleBuffer(b.data.data(), BYTE_SIZE(b.data), usage),
        };
    }

    VulkanBuffer createUnknownBuffer() {
        const std::vector<float> xZero(systemSize, 0.0f);
        constexpr auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        return device.createCpuVisibleBuffer(xZero.data(), BYTE_SIZE(xZero), usage);
    }

    VulkanBuffer solve(const CpuMatrix& A,
                       const CpuVector& b,
                       linalg::gpu::Solver::Method method,
                       uint32_t numIterations) {
        auto gpuSystem = createGpuSystem(A, b);
        auto unknown = createUnknownBuffer();

        linalg::gpu::Solver solver{device};
        solver.init(unknown.size);

        execute([&](auto commandBuffer) {
            solver.solve(commandBuffer, {
                .Coefficients = {
                    .values = gpuSystem.values,
                    .colIndices = gpuSystem.colIndices,
                    .rowOffsets = gpuSystem.rowOffsets,
                    .numRows = systemSize,
                    .numCols = systemSize,
                },
                .solution = gpuSystem.rhs,
                .unknown = unknown,
                .method = method,
                .numIterations = numIterations,
            });
        });

        return unknown;
    }

    void expectNear(const CpuVector& expected, const VulkanBuffer& actualBuffer, float tolerance) {
        auto actual = actualBuffer.span<float>(systemSize);
        for(uint32_t i = 0; i < systemSize; ++i) {
            ASSERT_NEAR(expected(i), actual[i], tolerance);
        }
        actualBuffer.unmap();
    }
};

TEST_F(LinearSolversTest, jacobiMatchesCpu) {
    const auto system = linear_system::load(dataPath("linear_system/euler_spd_16.txt"));
    ASSERT_EQ(systemSize, system.rows);
    ASSERT_EQ(systemSize, system.cols);
    ASSERT_EQ(systemSize, system.b.size());

    const auto A = linear_system::create_matrix<CpuMatrix>(system);
    const auto b = linear_system::create_vector<CpuVector>(system);

    const auto expected = linalg::jacobi(A, b, linalg::solver_options{
        .max_iterations = solverIterations,
        .tolerance = 0.0,
    });
    const auto actual = solve(A, b, linalg::gpu::Solver::Method::Jacobi, solverIterations);

    expectNear(expected.x, actual, 1e-4f);
}

TEST_F(LinearSolversTest, redBlackGaussSeidelMatchesCpu) {
    const auto system = linear_system::load(dataPath("linear_system/euler_spd_16.txt"));
    ASSERT_EQ(systemSize, system.rows);
    ASSERT_EQ(systemSize, system.cols);
    ASSERT_EQ(systemSize, system.b.size());

    const auto A = linear_system::create_matrix<CpuMatrix>(system);
    const auto b = linear_system::create_vector<CpuVector>(system);

    const auto expected = redBlackGaussSeidel(A, b, solverIterations);
    const auto actual = solve(A, b, linalg::gpu::Solver::Method::RedBlackGaussSeidel, solverIterations);

    expectNear(expected, actual, 2e-4f);
}

TEST_F(LinearSolversTest, conjugateGradientMatchesCpu) {
    const auto system = linear_system::load(dataPath("linear_system/euler_spd_16.txt"));
    ASSERT_EQ(systemSize, system.rows);
    ASSERT_EQ(systemSize, system.cols);
    ASSERT_EQ(systemSize, system.b.size());

    const auto A = linear_system::create_matrix<CpuMatrix>(system);
    const auto b = linear_system::create_vector<CpuVector>(system);

    const auto expected = linalg::cg(A, b, linalg::solver_options{
        .max_iterations = solverIterations,
        .tolerance = 1e-6,
    });
    const auto actual = solve(A, b, linalg::gpu::Solver::Method::ConjugateGradient, solverIterations);

    expectNear(expected.x, actual, 1e-3f);
}

TEST_F(LinearSolversTest, refreshesDescriptorSetWhenUnknownBufferChanges) {
    const auto system = linear_system::load(dataPath("linear_system/euler_spd_16.txt"));
    ASSERT_EQ(systemSize, system.rows);
    ASSERT_EQ(systemSize, system.cols);
    ASSERT_EQ(systemSize, system.b.size());

    const auto A = linear_system::create_matrix<CpuMatrix>(system);
    const auto b = linear_system::create_vector<CpuVector>(system);
    const auto expected = linalg::jacobi(A, b, linalg::solver_options{
        .max_iterations = solverIterations,
        .tolerance = 0.0,
    });

    auto gpuSystem = createGpuSystem(A, b);
    auto firstUnknown = createUnknownBuffer();
    auto secondUnknown = createUnknownBuffer();
    linalg::gpu::Solver solver{device};
    solver.init(firstUnknown.size);

    auto run = [&](VulkanBuffer& unknown) {
        execute([&](auto commandBuffer) {
            solver.solve(commandBuffer, {
                .Coefficients = {
                    .values = gpuSystem.values,
                    .colIndices = gpuSystem.colIndices,
                    .rowOffsets = gpuSystem.rowOffsets,
                    .numRows = systemSize,
                    .numCols = systemSize,
                },
                .solution = gpuSystem.rhs,
                .unknown = unknown,
                .method = linalg::gpu::Solver::Method::Jacobi,
                .numIterations = solverIterations,
            });
        });
    };

    run(firstUnknown);
    run(secondUnknown);

    expectNear(expected.x, firstUnknown, 1e-4f);
    expectNear(expected.x, secondUnknown, 1e-4f);
}
