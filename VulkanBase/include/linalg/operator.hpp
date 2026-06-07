#pragma once

#include "factory.hpp"

#include <cstddef>
#include <type_traits>

namespace linalg {
    template<typename MatrixType>
    concept cpu_matrix_type = requires {
        typename MatrixType::backend;
        typename MatrixType::layout;
        typename MatrixType::value_type;
        MatrixType::cols;
        MatrixType::matrix_layout;
        MatrixType::rows;
        MatrixType::storage;
    } && std::is_same_v<typename MatrixType::backend, cpu_backend>;

    template<typename MatrixType>
    concept cpu_dense_matrix_type = cpu_matrix_type<MatrixType> && MatrixType::storage == matrix_storage::dense;

    template<typename MatrixType>
    concept cpu_sparse_matrix_type = cpu_matrix_type<MatrixType> && MatrixType::storage == matrix_storage::sparse;

    template<typename MatrixType>
    concept cpu_dense_vector_type =
        cpu_dense_matrix_type<MatrixType> && std::is_same_v<typename MatrixType::layout, vector_tag>;

    template<typename MatrixType, typename ValueType>
    using rebind_matrix_value_t =
        matrix<ValueType, MatrixType::rows, MatrixType::cols, typename MatrixType::layout, typename MatrixType::backend>;

    template<typename RhsMatrixType, typename ValueType, size_t Rows>
    using product_column_result_t = std::conditional_t<cpu_dense_vector_type<RhsMatrixType>,
                                                       vector<ValueType, Rows>,
                                                       dense_row_matrix<ValueType, Rows, RhsMatrixType::cols>>;

    template<typename MatrixType>
        requires cpu_dense_matrix_type<MatrixType>
    auto transpose(const MatrixType& matrix) {
        using value_type = typename MatrixType::value_type;
        using result_type = dense_row_matrix<value_type, MatrixType::cols, MatrixType::rows>;

        auto result = create_matrix<result_type>();

        for (size_t r = 0; r < MatrixType::rows; ++r) {
            for (size_t c = 0; c < MatrixType::cols; ++c) {
                result(c, r) = matrix(r, c);
            }
        }

        return result;
    }

    template<typename LhsMatrixType, typename RhsMatrixType>
        requires cpu_dense_matrix_type<LhsMatrixType> && cpu_dense_matrix_type<RhsMatrixType> &&
                 (LhsMatrixType::rows == RhsMatrixType::rows) && (LhsMatrixType::cols == RhsMatrixType::cols)
    auto operator+(const LhsMatrixType& lhs, const RhsMatrixType& rhs) {
        using value_type = std::common_type_t<typename LhsMatrixType::value_type, typename RhsMatrixType::value_type>;
        using result_type = rebind_matrix_value_t<LhsMatrixType, value_type>;

        auto result = create_matrix<result_type>();

        for (size_t r = 0; r < LhsMatrixType::rows; ++r) {
            for (size_t c = 0; c < LhsMatrixType::cols; ++c) {
                result(r, c) = static_cast<value_type>(lhs(r, c)) + static_cast<value_type>(rhs(r, c));
            }
        }

        return result;
    }

    template<typename MatrixType>
        requires cpu_dense_matrix_type<MatrixType>
    MatrixType& operator+=(MatrixType& lhs, const MatrixType& rhs) {
        for (size_t r = 0; r < MatrixType::rows; ++r) {
            for (size_t c = 0; c < MatrixType::cols; ++c) {
                lhs(r, c) += rhs(r, c);
            }
        }

        return lhs;
    }

    template<typename LhsMatrixType, typename RhsMatrixType>
        requires cpu_dense_matrix_type<LhsMatrixType> && cpu_dense_matrix_type<RhsMatrixType> &&
                 (LhsMatrixType::rows == RhsMatrixType::rows) && (LhsMatrixType::cols == RhsMatrixType::cols)
    auto operator-(const LhsMatrixType& lhs, const RhsMatrixType& rhs) {
        using value_type = std::common_type_t<typename LhsMatrixType::value_type, typename RhsMatrixType::value_type>;
        using result_type = rebind_matrix_value_t<LhsMatrixType, value_type>;

        auto result = create_matrix<result_type>();

        for (size_t r = 0; r < LhsMatrixType::rows; ++r) {
            for (size_t c = 0; c < LhsMatrixType::cols; ++c) {
                result(r, c) = static_cast<value_type>(lhs(r, c)) - static_cast<value_type>(rhs(r, c));
            }
        }

        return result;
    }

    template<typename MatrixType>
        requires cpu_dense_matrix_type<MatrixType>
    MatrixType& operator-=(MatrixType& lhs, const MatrixType& rhs) {
        for (size_t r = 0; r < MatrixType::rows; ++r) {
            for (size_t c = 0; c < MatrixType::cols; ++c) {
                lhs(r, c) -= rhs(r, c);
            }
        }

        return lhs;
    }

    template<typename MatrixType>
        requires cpu_matrix_type<MatrixType>
    MatrixType operator-(const MatrixType& matrix) {
        auto result = matrix;

        for (auto& value : result.data) {
            value = -value;
        }

        return result;
    }

    template<typename MatrixType, typename Scalar>
        requires cpu_matrix_type<MatrixType> && std::is_convertible_v<Scalar, typename MatrixType::value_type>
    MatrixType operator*(const MatrixType& matrix, Scalar scalar) {
        auto result = matrix;
        const auto value = static_cast<typename MatrixType::value_type>(scalar);

        for (auto& element : result.data) {
            element *= value;
        }

        return result;
    }

    template<typename Scalar, typename MatrixType>
        requires cpu_matrix_type<MatrixType> && std::is_convertible_v<Scalar, typename MatrixType::value_type>
    MatrixType operator*(Scalar scalar, const MatrixType& matrix) {
        return matrix * scalar;
    }

    template<typename MatrixType, typename Scalar>
        requires cpu_matrix_type<MatrixType> && std::is_convertible_v<Scalar, typename MatrixType::value_type>
    MatrixType& operator*=(MatrixType& matrix, Scalar scalar) {
        const auto value = static_cast<typename MatrixType::value_type>(scalar);

        for (auto& element : matrix.data) {
            element *= value;
        }

        return matrix;
    }

    template<typename MatrixType, typename Scalar>
        requires cpu_matrix_type<MatrixType> && std::is_convertible_v<Scalar, typename MatrixType::value_type>
    MatrixType operator/(const MatrixType& matrix, Scalar scalar) {
        auto result = matrix;
        const auto value = static_cast<typename MatrixType::value_type>(scalar);

        for (auto& element : result.data) {
            element /= value;
        }

        return result;
    }

    template<typename MatrixType, typename Scalar>
        requires cpu_matrix_type<MatrixType> && std::is_convertible_v<Scalar, typename MatrixType::value_type>
    MatrixType& operator/=(MatrixType& matrix, Scalar scalar) {
        const auto value = static_cast<typename MatrixType::value_type>(scalar);

        for (auto& element : matrix.data) {
            element /= value;
        }

        return matrix;
    }

    template<typename LhsVectorType, typename RhsVectorType>
        requires cpu_dense_vector_type<LhsVectorType> && cpu_dense_vector_type<RhsVectorType> &&
                 (LhsVectorType::rows == RhsVectorType::rows)
    auto dot(const LhsVectorType& lhs, const RhsVectorType& rhs) {
        using value_type = std::common_type_t<typename LhsVectorType::value_type, typename RhsVectorType::value_type>;

        value_type result{};

        for (size_t r = 0; r < LhsVectorType::rows; ++r) {
            result += static_cast<value_type>(lhs(r, 0)) * static_cast<value_type>(rhs(r, 0));
        }

        return result;
    }

    template<typename LhsVectorType, typename RhsVectorType>
        requires cpu_dense_vector_type<LhsVectorType> && cpu_dense_vector_type<RhsVectorType> &&
                 (LhsVectorType::rows == 3) && (RhsVectorType::rows == 3)
    auto cross(const LhsVectorType& lhs, const RhsVectorType& rhs) {
        using value_type = std::common_type_t<typename LhsVectorType::value_type, typename RhsVectorType::value_type>;
        using result_type = vector<value_type, 3>;

        auto result = create_matrix<result_type>();

        result(0) = static_cast<value_type>(lhs(1)) * static_cast<value_type>(rhs(2)) -
                    static_cast<value_type>(lhs(2)) * static_cast<value_type>(rhs(1));
        result(1) = static_cast<value_type>(lhs(2)) * static_cast<value_type>(rhs(0)) -
                    static_cast<value_type>(lhs(0)) * static_cast<value_type>(rhs(2));
        result(2) = static_cast<value_type>(lhs(0)) * static_cast<value_type>(rhs(1)) -
                    static_cast<value_type>(lhs(1)) * static_cast<value_type>(rhs(0));

        return result;
    }

    template<typename LhsVectorType, typename RhsVectorType>
        requires cpu_dense_vector_type<LhsVectorType> && cpu_dense_vector_type<RhsVectorType> &&
                 (LhsVectorType::rows == RhsVectorType::rows)
    auto operator*(const LhsVectorType& lhs, const RhsVectorType& rhs) {
        return dot(lhs, rhs);
    }

    template<typename LhsMatrixType, typename RhsMatrixType>
        requires cpu_dense_matrix_type<LhsMatrixType> && cpu_dense_matrix_type<RhsMatrixType> &&
                 (LhsMatrixType::cols == RhsMatrixType::rows) &&
                 !(cpu_dense_vector_type<LhsMatrixType> && cpu_dense_vector_type<RhsMatrixType> &&
                   LhsMatrixType::rows == RhsMatrixType::rows)
    auto operator*(const LhsMatrixType& lhs, const RhsMatrixType& rhs) {
        using value_type = std::common_type_t<typename LhsMatrixType::value_type, typename RhsMatrixType::value_type>;
        using result_type = product_column_result_t<RhsMatrixType, value_type, LhsMatrixType::rows>;

        auto result = create_matrix<result_type>();

        for (size_t r = 0; r < LhsMatrixType::rows; ++r) {
            for (size_t c = 0; c < RhsMatrixType::cols; ++c) {
                value_type sum{};

                for (size_t k = 0; k < LhsMatrixType::cols; ++k) {
                    sum += static_cast<value_type>(lhs(r, k)) * static_cast<value_type>(rhs(k, c));
                }

                result(r, c) = sum;
            }
        }

        return result;
    }

    template<typename LhsMatrixType, typename RhsMatrixType>
        requires cpu_sparse_matrix_type<LhsMatrixType> && cpu_dense_matrix_type<RhsMatrixType> &&
                 (LhsMatrixType::cols == RhsMatrixType::rows)
    auto operator*(const LhsMatrixType& lhs, const RhsMatrixType& rhs) {
        using value_type = std::common_type_t<typename LhsMatrixType::value_type, typename RhsMatrixType::value_type>;
        using result_type = product_column_result_t<RhsMatrixType, value_type, LhsMatrixType::rows>;

        auto result = create_matrix<result_type>();

        if constexpr (LhsMatrixType::matrix_layout == matrix_layout_type::csr) {
            for (size_t r = 0; r < LhsMatrixType::rows; ++r) {
                const auto start = lhs.row_offsets[r];
                const auto end = lhs.row_offsets[r + 1];

                for (auto i = start; i < end; ++i) {
                    const auto k = lhs.col_indices[i];
                    const auto lhs_value = static_cast<value_type>(lhs.data[i]);

                    for (size_t c = 0; c < RhsMatrixType::cols; ++c) {
                        result(r, c) += lhs_value * static_cast<value_type>(rhs(k, c));
                    }
                }
            }
        } else if constexpr (LhsMatrixType::matrix_layout == matrix_layout_type::csc) {
            for (size_t k = 0; k < LhsMatrixType::cols; ++k) {
                const auto start = lhs.col_offsets[k];
                const auto end = lhs.col_offsets[k + 1];

                for (auto i = start; i < end; ++i) {
                    const auto r = lhs.row_indices[i];
                    const auto lhs_value = static_cast<value_type>(lhs.data[i]);

                    for (size_t c = 0; c < RhsMatrixType::cols; ++c) {
                        result(r, c) += lhs_value * static_cast<value_type>(rhs(k, c));
                    }
                }
            }
        }

        return result;
    }

    template<typename LhsMatrixType, typename RhsMatrixType>
        requires cpu_dense_matrix_type<LhsMatrixType> && cpu_sparse_matrix_type<RhsMatrixType> &&
                 (LhsMatrixType::cols == RhsMatrixType::rows)
    auto operator*(const LhsMatrixType& lhs, const RhsMatrixType& rhs) {
        using value_type = std::common_type_t<typename LhsMatrixType::value_type, typename RhsMatrixType::value_type>;
        using result_type = dense_row_matrix<value_type, LhsMatrixType::rows, RhsMatrixType::cols>;

        auto result = create_matrix<result_type>();

        if constexpr (RhsMatrixType::matrix_layout == matrix_layout_type::csr) {
            for (size_t k = 0; k < RhsMatrixType::rows; ++k) {
                const auto start = rhs.row_offsets[k];
                const auto end = rhs.row_offsets[k + 1];

                for (auto i = start; i < end; ++i) {
                    const auto c = rhs.col_indices[i];
                    const auto rhs_value = static_cast<value_type>(rhs.data[i]);

                    for (size_t r = 0; r < LhsMatrixType::rows; ++r) {
                        result(r, c) += static_cast<value_type>(lhs(r, k)) * rhs_value;
                    }
                }
            }
        } else if constexpr (RhsMatrixType::matrix_layout == matrix_layout_type::csc) {
            for (size_t c = 0; c < RhsMatrixType::cols; ++c) {
                const auto start = rhs.col_offsets[c];
                const auto end = rhs.col_offsets[c + 1];

                for (auto i = start; i < end; ++i) {
                    const auto k = rhs.row_indices[i];
                    const auto rhs_value = static_cast<value_type>(rhs.data[i]);

                    for (size_t r = 0; r < LhsMatrixType::rows; ++r) {
                        result(r, c) += static_cast<value_type>(lhs(r, k)) * rhs_value;
                    }
                }
            }
        }

        return result;
    }

    template<typename LhsMatrixType, typename RhsMatrixType>
        requires cpu_dense_matrix_type<LhsMatrixType> && cpu_dense_matrix_type<RhsMatrixType> &&
                 (LhsMatrixType::rows == RhsMatrixType::rows) && (LhsMatrixType::cols == RhsMatrixType::cols)
    bool operator==(const LhsMatrixType& lhs, const RhsMatrixType& rhs) {
        for (size_t r = 0; r < LhsMatrixType::rows; ++r) {
            for (size_t c = 0; c < LhsMatrixType::cols; ++c) {
                if (lhs(r, c) != rhs(r, c)) {
                    return false;
                }
            }
        }

        return true;
    }

    template<typename LhsMatrixType, typename RhsMatrixType>
        requires cpu_dense_matrix_type<LhsMatrixType> && cpu_dense_matrix_type<RhsMatrixType> &&
                 (LhsMatrixType::rows == RhsMatrixType::rows) && (LhsMatrixType::cols == RhsMatrixType::cols)
    bool operator!=(const LhsMatrixType& lhs, const RhsMatrixType& rhs) {
        return !(lhs == rhs);
    }
}
