#pragma once

#include "matrix.hpp"
#include "matrix_market.hpp"

#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace linalg {
    enum class input_layout { row_major, column_major };

    template<typename MatrixType>
        requires std::is_same_v<typename MatrixType::backend, cpu_backend>
    MatrixType create_matrix() {
        MatrixType m{};

        if constexpr (MatrixType::storage == matrix_storage::dense) {
            m.data.resize(MatrixType::rows * MatrixType::cols);
        } else if constexpr (MatrixType::matrix_layout == matrix_layout_type::csr) {
            m.row_offsets.resize(MatrixType::rows + 1);
        } else if constexpr (MatrixType::matrix_layout == matrix_layout_type::csc) {
            m.col_offsets.resize(MatrixType::cols + 1);
        }

        return m;
    }

    template<typename MatrixType>
        requires std::is_same_v<typename MatrixType::backend, cpu_backend>
    MatrixType create_from_dense_matrix(std::span<const typename MatrixType::value_type> data,
                                        input_layout source_layout = input_layout::row_major) {
        using T = typename MatrixType::value_type;

        MatrixType m = create_matrix<MatrixType>();

        if (data.size() != MatrixType::rows * MatrixType::cols) {
            throw std::runtime_error{"Invalid dense input size"};
        }

        if constexpr (MatrixType::storage == matrix_storage::dense) {
            for (size_t r = 0; r < MatrixType::rows; ++r) {
                for (size_t c = 0; c < MatrixType::cols; ++c) {
                    const size_t src = source_layout == input_layout::row_major
                                           ? r * MatrixType::cols + c
                                           : c * MatrixType::rows + r;
                    const size_t dst = MatrixType::matrix_layout == matrix_layout_type::row_major
                                           ? r * MatrixType::cols + c
                                           : c * MatrixType::rows + r;
                    m.data[dst] = data[src];
                }
            }
        } else if constexpr (MatrixType::matrix_layout == matrix_layout_type::csr) {
            m.row_offsets.assign(MatrixType::rows + 1, 0);

            for (size_t r = 0; r < MatrixType::rows; ++r) {
                m.row_offsets[r] = static_cast<uint32_t>(m.data.size());

                for (size_t c = 0; c < MatrixType::cols; ++c) {
                    const size_t src = source_layout == input_layout::row_major
                                           ? r * MatrixType::cols + c
                                           : c * MatrixType::rows + r;

                    if (data[src] != T{}) {
                        m.data.push_back(data[src]);
                        m.col_indices.push_back(static_cast<uint32_t>(c));
                    }
                }
            }

            m.row_offsets[MatrixType::rows] = static_cast<uint32_t>(m.data.size());
        } else if constexpr (MatrixType::matrix_layout == matrix_layout_type::csc) {
            m.col_offsets.assign(MatrixType::cols + 1, 0);

            for (size_t c = 0; c < MatrixType::cols; ++c) {
                m.col_offsets[c] = static_cast<uint32_t>(m.data.size());

                for (size_t r = 0; r < MatrixType::rows; ++r) {
                    const size_t src = source_layout == input_layout::row_major
                                           ? r * MatrixType::cols + c
                                           : c * MatrixType::rows + r;

                    if (data[src] != T{}) {
                        m.data.push_back(data[src]);
                        m.row_indices.push_back(static_cast<uint32_t>(r));
                    }
                }
            }

            m.col_offsets[MatrixType::cols] = static_cast<uint32_t>(m.data.size());
        }

        return m;
    }

    template<typename MatrixType>
        requires std::is_same_v<typename MatrixType::backend, cpu_backend>
    MatrixType create_matrix(const matrix_market::matrix& source) {
        using T = typename MatrixType::value_type;

        MatrixType m = create_matrix<MatrixType>();

        if (source.rowCount != MatrixType::rows || source.colCount != MatrixType::cols) {
            throw std::runtime_error{"dimensions don't match"};
        }

        if constexpr (MatrixType::storage == matrix_storage::dense) {
            if constexpr (MatrixType::matrix_layout == matrix_layout_type::row_major) {
                for (const auto [row, col, value] : source) {
                    m(row, col) = value;
                }
            } else {
                for (const auto [row, col, value] : source) {
                    m(col, row) = value;
                }
            }
        } else if constexpr (MatrixType::matrix_layout == matrix_layout_type::csr) {
            const auto num_elements = source.rowCount * source.colCount;
            std::vector<uint32_t> flags(num_elements, 0);
            std::vector<uint32_t> offsets(num_elements);

            for (auto [row, col, _] : source) {
                auto index = row * source.colCount + col;
                flags[index] = 1;
            }

            std::exclusive_scan(flags.begin(), flags.end(), offsets.begin(), 0u);
            const auto non_zero_count = flags.back() + offsets.back();
            m.data.resize(non_zero_count);
            m.col_indices.resize(non_zero_count);

            for (auto i = 0; i < num_elements; ++i) {
                if (flags[i] == 1) {
                    auto row = i / source.colCount;
                    auto col = i % source.colCount;
                    auto dst = offsets[i];

                    auto value = source(row, col);
                    m.data[dst] = static_cast<T>(value);
                    m.col_indices[dst] = col;
                }
            }

            std::vector<uint32_t> row_counts(source.rowCount);

            for (auto r = 0; r < source.rowCount; ++r) {
                row_counts[r] = source.row(r).size();
            }

            m.row_offsets.resize(source.rowCount + 1);
            std::exclusive_scan(row_counts.begin(), row_counts.end(), m.row_offsets.begin(), 0u);
            m.row_offsets.back() = non_zero_count;
        } else if constexpr (MatrixType::matrix_layout == matrix_layout_type::csc) {
            throw std::runtime_error{"csc not yet implemented"};
        }

        return m;
    }

    template<typename MatrixType>
        requires std::is_same_v<typename MatrixType::backend, vulkan_backend> &&
                 (MatrixType::storage == matrix_storage::dense)
    MatrixType create_matrix(const VulkanDevice& device,
                             VkBufferUsageFlags buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_GPU_ONLY) {
        MatrixType m{};
        const auto size = MatrixType::rows * MatrixType::cols * sizeof(typename MatrixType::value_type);
        m.data = device.createBuffer(buffer_usage, memory_usage, size);
        return m;
    }

    template<typename MatrixType>
        requires std::is_same_v<typename MatrixType::backend, vulkan_backend> &&
                 (MatrixType::storage == matrix_storage::sparse)
    MatrixType create_matrix(const VulkanDevice& device,
                             size_t max_non_zero_count,
                             VkBufferUsageFlags buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_GPU_ONLY) {
        MatrixType m{};

        const auto value_size = sizeof(typename MatrixType::value_type) * max_non_zero_count;
        const auto index_size = sizeof(uint32_t) * max_non_zero_count;
        m.data = device.createBuffer(buffer_usage, memory_usage, value_size);

        if constexpr (MatrixType::matrix_layout == matrix_layout_type::csr) {
            const auto offset_size = (MatrixType::rows + 1) * sizeof(uint32_t);
            m.col_indices = device.createBuffer(buffer_usage, memory_usage, index_size);
            m.row_offsets = device.createBuffer(buffer_usage, memory_usage, offset_size);
        } else if constexpr (MatrixType::matrix_layout == matrix_layout_type::csc) {
            const auto offset_size = (MatrixType::cols + 1) * sizeof(uint32_t);
            m.row_indices = device.createBuffer(buffer_usage, memory_usage, index_size);
            m.col_offsets = device.createBuffer(buffer_usage, memory_usage, offset_size);
        }

        return m;
    }
}
