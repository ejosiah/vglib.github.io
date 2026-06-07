#pragma once

#include "backend.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace linalg {
    enum class matrix_storage { dense, sparse };

    enum class matrix_layout_type { row_major, column_major, csr, csc };

    struct row_major_tag {};

    struct column_major_tag {};

    struct vector_tag {};

    struct csr_tag {};

    struct csc_tag {};

    template<typename T, size_t Rows, size_t Cols, typename LayoutTag, typename Backend = cpu_backend>
    struct matrix;

    template<typename T, size_t Rows, size_t Cols, typename Backend>
    struct matrix<T, Rows, Cols, row_major_tag, Backend> {
        using value_type = T;
        using layout = row_major_tag;
        using backend = Backend;

        static constexpr size_t rows = Rows;
        static constexpr size_t cols = Cols;

        static constexpr matrix_storage storage = matrix_storage::dense;
        static constexpr matrix_layout_type matrix_layout = matrix_layout_type::row_major;

        buffer_type_t<Backend, T> data;

        T& operator()(size_t r, size_t c)
        requires std::is_same_v<Backend, cpu_backend> {
            return data[r * Cols + c];
        }

        const T& operator()(size_t r, size_t c) const
        requires std::is_same_v<Backend, cpu_backend> {
            return data[r * Cols + c];
        }
    };

    template<typename T, size_t Rows, size_t Cols, typename Backend>
    struct matrix<T, Rows, Cols, column_major_tag, Backend> {
        using value_type = T;
        using layout = column_major_tag;
        using backend = Backend;

        static constexpr size_t rows = Rows;
        static constexpr size_t cols = Cols;

        static constexpr matrix_storage storage = matrix_storage::dense;
        static constexpr matrix_layout_type matrix_layout = matrix_layout_type::column_major;

        buffer_type_t<Backend, T> data;

        T& operator()(size_t r, size_t c)
        requires std::is_same_v<Backend, cpu_backend> {
            return data[c * Rows + r];
        }

        const T& operator()(size_t r, size_t c) const
        requires std::is_same_v<Backend, cpu_backend> {
            return data[c * Rows + r];
        }
    };

    template<typename T, size_t Rows, typename Backend>
    struct matrix<T, Rows, 1, vector_tag, Backend> {
        using value_type = T;
        using layout = vector_tag;
        using backend = Backend;

        static constexpr size_t rows = Rows;
        static constexpr size_t cols = 1;

        static constexpr matrix_storage storage = matrix_storage::dense;
        static constexpr matrix_layout_type matrix_layout = matrix_layout_type::row_major;

        buffer_type_t<Backend, T> data;

        T& operator()(size_t r)
        requires std::is_same_v<Backend, cpu_backend> {
            return data[r];
        }

        const T& operator()(size_t r) const
        requires std::is_same_v<Backend, cpu_backend> {
            return data[r];
        }

        T& operator()(size_t r, size_t c)
        requires std::is_same_v<Backend, cpu_backend> {
            return data[r * cols + c];
        }

        const T& operator()(size_t r, size_t c) const
        requires std::is_same_v<Backend, cpu_backend> {
            return data[r * cols + c];
        }
    };

    template<typename T, size_t Rows, size_t Cols, typename Backend>
    struct matrix<T, Rows, Cols, csr_tag, Backend> {
        using value_type = T;
        using layout = csr_tag;
        using backend = Backend;

        static constexpr size_t rows = Rows;
        static constexpr size_t cols = Cols;

        static constexpr matrix_storage storage = matrix_storage::sparse;
        static constexpr matrix_layout_type matrix_layout = matrix_layout_type::csr;

        buffer_type_t<Backend, T> data;
        buffer_type_t<Backend, uint32_t> col_indices;
        buffer_type_t<Backend, uint32_t> row_offsets;
    };

    template<typename T, size_t Rows, size_t Cols, typename Backend>
    struct matrix<T, Rows, Cols, csc_tag, Backend> {
        using value_type = T;
        using layout = csc_tag;
        using backend = Backend;

        static constexpr size_t rows = Rows;
        static constexpr size_t cols = Cols;

        static constexpr matrix_storage storage = matrix_storage::sparse;
        static constexpr matrix_layout_type matrix_layout = matrix_layout_type::csc;

        buffer_type_t<Backend, T> data;
        buffer_type_t<Backend, uint32_t> row_indices;
        buffer_type_t<Backend, uint32_t> col_offsets;
    };

    template<typename M>
    inline constexpr bool is_dense_v = M::storage == matrix_storage::dense;

    template<typename M>
    inline constexpr bool is_sparse_v = M::storage == matrix_storage::sparse;

    template<typename M>
    inline constexpr bool is_row_major_v = M::matrix_layout == matrix_layout_type::row_major;

    template<typename M>
    inline constexpr bool is_csr_v = M::matrix_layout == matrix_layout_type::csr;

    template<typename T, size_t Rows, size_t Cols, typename Backend = cpu_backend>
    using dense_row_matrix = matrix<T, Rows, Cols, row_major_tag, Backend>;

    template<typename T, size_t Rows, size_t Cols, typename Backend = cpu_backend>
    using dense_column_matrix = matrix<T, Rows, Cols, column_major_tag, Backend>;

    template<typename T, size_t Rows, size_t Cols, typename Backend = cpu_backend>
    using sparse_csr_matrix = matrix<T, Rows, Cols, csr_tag, Backend>;

    template<typename T, size_t Rows, size_t Cols, typename Backend = cpu_backend>
    using sparse_csc_matrix = matrix<T, Rows, Cols, csc_tag, Backend>;

    template<typename T, size_t Rows, typename Backend = cpu_backend>
    using vector = matrix<T, Rows, 1, vector_tag, Backend>;
}
