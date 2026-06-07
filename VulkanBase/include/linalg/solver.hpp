#pragma once

#include "operator.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace linalg {
    struct solver_options {
        size_t max_iterations{1000};
        double tolerance{1e-6};
    };

    template<typename VectorType>
    struct solver_result {
        VectorType x;
        size_t iterations{};
        double residual_norm{};
        bool converged{};
    };

    namespace detail {
        template<typename MatrixType>
            requires cpu_matrix_type<MatrixType>
        typename MatrixType::value_type matrix_value(const MatrixType& A, size_t row, size_t col);
    }

    struct identity_preconditioner {
        template<typename MatrixType, typename VectorType>
        void pre_step(const MatrixType&, const VectorType&, const VectorType&) {}

        template<typename VectorType>
        VectorType apply(const VectorType& residual) const {
            return residual;
        }
    };

    struct jacobi_preconditioner {
        std::vector<double> inverse_diagonal;

        template<typename MatrixType, typename VectorType>
            requires cpu_matrix_type<MatrixType> && cpu_dense_vector_type<VectorType> &&
                     (MatrixType::rows == MatrixType::cols) && (MatrixType::rows == VectorType::rows)
        void pre_step(const MatrixType& A, const VectorType&, const VectorType&) {
            inverse_diagonal.resize(MatrixType::rows);

            for (size_t i = 0; i < MatrixType::rows; ++i) {
                const auto diagonal = detail::matrix_value(A, i, i);

                if (diagonal == typename MatrixType::value_type{}) {
                    throw std::runtime_error{"zero diagonal in jacobi preconditioner"};
                }

                inverse_diagonal[i] = 1.0 / static_cast<double>(diagonal);
            }
        }

        template<typename VectorType>
            requires cpu_dense_vector_type<VectorType>
        VectorType apply(const VectorType& residual) const {
            auto result = create_matrix<VectorType>();

            if (inverse_diagonal.size() != VectorType::rows) {
                throw std::runtime_error{"jacobi preconditioner was not initialized"};
            }

            for (size_t i = 0; i < VectorType::rows; ++i) {
                result(i) = static_cast<typename VectorType::value_type>(
                    static_cast<double>(residual(i)) * inverse_diagonal[i]);
            }

            return result;
        }
    };

    namespace detail {
        template<typename MatrixType>
            requires cpu_matrix_type<MatrixType>
        typename MatrixType::value_type matrix_value(const MatrixType& A, size_t row, size_t col) {
            if constexpr (cpu_dense_matrix_type<MatrixType>) {
                return A(row, col);
            } else if constexpr (MatrixType::matrix_layout == matrix_layout_type::csr) {
                const auto start = A.row_offsets[row];
                const auto end = A.row_offsets[row + 1];

                for (auto i = start; i < end; ++i) {
                    if (A.col_indices[i] == col) {
                        return A.data[i];
                    }
                }
            } else if constexpr (MatrixType::matrix_layout == matrix_layout_type::csc) {
                const auto start = A.col_offsets[col];
                const auto end = A.col_offsets[col + 1];

                for (auto i = start; i < end; ++i) {
                    if (A.row_indices[i] == row) {
                        return A.data[i];
                    }
                }
            }

            return {};
        }

        template<typename MatrixType, typename Func>
            requires cpu_matrix_type<MatrixType>
        void for_each_row_entry(const MatrixType& A, size_t row, Func&& func) {
            if constexpr (cpu_dense_matrix_type<MatrixType>) {
                for (size_t col = 0; col < MatrixType::cols; ++col) {
                    func(col, A(row, col));
                }
            } else if constexpr (MatrixType::matrix_layout == matrix_layout_type::csr) {
                const auto start = A.row_offsets[row];
                const auto end = A.row_offsets[row + 1];

                for (auto i = start; i < end; ++i) {
                    func(static_cast<size_t>(A.col_indices[i]), A.data[i]);
                }
            } else if constexpr (MatrixType::matrix_layout == matrix_layout_type::csc) {
                for (size_t col = 0; col < MatrixType::cols; ++col) {
                    const auto start = A.col_offsets[col];
                    const auto end = A.col_offsets[col + 1];

                    for (auto i = start; i < end; ++i) {
                        if (A.row_indices[i] == row) {
                            func(col, A.data[i]);
                        }
                    }
                }
            }
        }

        template<typename VectorType>
            requires cpu_dense_vector_type<VectorType>
        double norm(const VectorType& v) {
            return std::sqrt(static_cast<double>(dot(v, v)));
        }

        template<typename Preconditioner, typename MatrixType, typename VectorType>
        void preconditioner_pre_step(Preconditioner& preconditioner,
                                     const MatrixType& A,
                                     const VectorType& b,
                                     const VectorType& x) {
            if constexpr (requires { preconditioner.pre_step(A, b, x); }) {
                preconditioner.pre_step(A, b, x);
            }
        }

        template<typename Preconditioner, typename VectorType>
        VectorType apply_preconditioner(Preconditioner& preconditioner, const VectorType& residual) {
            if constexpr (requires { preconditioner.apply(residual); }) {
                return preconditioner.apply(residual);
            } else {
                return preconditioner(residual);
            }
        }

        template<typename MatrixType, typename VectorType>
        concept linear_system_type =
            cpu_matrix_type<MatrixType> && cpu_dense_vector_type<VectorType> &&
            (MatrixType::rows == MatrixType::cols) &&
            (MatrixType::rows == VectorType::rows);

        template<typename Preconditioner>
        concept preconditioner_argument =
            !std::is_same_v<std::remove_cvref_t<Preconditioner>, solver_options>;
    }

    template<typename MatrixType, typename VectorType>
        requires detail::linear_system_type<MatrixType, VectorType>
    solver_result<VectorType> jacobi(const MatrixType& A,
                                     const VectorType& b,
                                     VectorType x,
                                     solver_options options = {}) {
        auto next = create_matrix<VectorType>();
        auto residual = b - A * x;
        auto residual_norm = detail::norm(residual);

        if (residual_norm <= options.tolerance) {
            return {x, 0, residual_norm, true};
        }

        for (size_t iteration = 0; iteration < options.max_iterations; ++iteration) {
            for (size_t row = 0; row < MatrixType::rows; ++row) {
                typename VectorType::value_type diagonal{};
                typename VectorType::value_type sum{};

                detail::for_each_row_entry(A, row, [&](size_t col, auto value) {
                    if (col == row) {
                        diagonal = static_cast<typename VectorType::value_type>(value);
                    } else {
                        sum += static_cast<typename VectorType::value_type>(value) * x(col);
                    }
                });

                if (diagonal == typename VectorType::value_type{}) {
                    throw std::runtime_error{"zero diagonal in jacobi solver"};
                }

                next(row) = (b(row) - sum) / diagonal;
            }

            x = next;
            residual = b - A * x;
            residual_norm = detail::norm(residual);

            if (residual_norm <= options.tolerance) {
                return {x, iteration + 1, residual_norm, true};
            }
        }

        return {x, options.max_iterations, residual_norm, false};
    }

    template<typename MatrixType, typename VectorType>
        requires detail::linear_system_type<MatrixType, VectorType>
    solver_result<VectorType> jacobi(const MatrixType& A,
                                     const VectorType& b,
                                     solver_options options = {}) {
        return jacobi(A, b, create_matrix<VectorType>(), options);
    }

    template<typename MatrixType, typename VectorType>
        requires detail::linear_system_type<MatrixType, VectorType>
    solver_result<VectorType> gauss_seidel(const MatrixType& A,
                                           const VectorType& b,
                                           VectorType x,
                                           solver_options options = {}) {
        auto residual = b - A * x;
        auto residual_norm = detail::norm(residual);

        if (residual_norm <= options.tolerance) {
            return {x, 0, residual_norm, true};
        }

        for (size_t iteration = 0; iteration < options.max_iterations; ++iteration) {
            for (size_t row = 0; row < MatrixType::rows; ++row) {
                typename VectorType::value_type diagonal{};
                typename VectorType::value_type sum{};

                detail::for_each_row_entry(A, row, [&](size_t col, auto value) {
                    if (col == row) {
                        diagonal = static_cast<typename VectorType::value_type>(value);
                    } else {
                        sum += static_cast<typename VectorType::value_type>(value) * x(col);
                    }
                });

                if (diagonal == typename VectorType::value_type{}) {
                    throw std::runtime_error{"zero diagonal in gauss-seidel solver"};
                }

                x(row) = (b(row) - sum) / diagonal;
            }

            residual = b - A * x;
            residual_norm = detail::norm(residual);

            if (residual_norm <= options.tolerance) {
                return {x, iteration + 1, residual_norm, true};
            }
        }

        return {x, options.max_iterations, residual_norm, false};
    }

    template<typename MatrixType, typename VectorType>
        requires detail::linear_system_type<MatrixType, VectorType>
    solver_result<VectorType> gauss_seidel(const MatrixType& A,
                                           const VectorType& b,
                                           solver_options options = {}) {
        return gauss_seidel(A, b, create_matrix<VectorType>(), options);
    }

    template<typename MatrixType, typename VectorType>
        requires detail::linear_system_type<MatrixType, VectorType>
    solver_result<VectorType> gs(const MatrixType& A,
                                 const VectorType& b,
                                 VectorType x,
                                 solver_options options = {}) {
        return gauss_seidel(A, b, x, options);
    }

    template<typename MatrixType, typename VectorType>
        requires detail::linear_system_type<MatrixType, VectorType>
    solver_result<VectorType> gs(const MatrixType& A,
                                 const VectorType& b,
                                 solver_options options = {}) {
        return gauss_seidel(A, b, options);
    }

    template<typename MatrixType, typename VectorType, typename Preconditioner>
        requires detail::linear_system_type<MatrixType, VectorType> &&
                 detail::preconditioner_argument<Preconditioner>
    solver_result<VectorType> pcg(const MatrixType& A,
                                  const VectorType& b,
                                  VectorType x,
                                  Preconditioner preconditioner,
                                  solver_options options = {}) {
        detail::preconditioner_pre_step(preconditioner, A, b, x);

        auto r = b - A * x;
        auto z = detail::apply_preconditioner(preconditioner, r);
        auto p = z;
        auto rz_old = dot(r, z);
        auto residual_norm = detail::norm(r);

        if (residual_norm <= options.tolerance) {
            return {x, 0, residual_norm, true};
        }

        for (size_t iteration = 0; iteration < options.max_iterations; ++iteration) {
            const auto Ap = A * p;
            const auto denominator = dot(p, Ap);

            if (std::abs(static_cast<double>(denominator)) <= std::numeric_limits<double>::epsilon()) {
                return {x, iteration, residual_norm, false};
            }

            const auto alpha = rz_old / denominator;
            x += alpha * p;
            r -= alpha * Ap;
            residual_norm = detail::norm(r);

            if (residual_norm <= options.tolerance) {
                return {x, iteration + 1, residual_norm, true};
            }

            z = detail::apply_preconditioner(preconditioner, r);
            const auto rz_new = dot(r, z);

            if (std::abs(static_cast<double>(rz_old)) <= std::numeric_limits<double>::epsilon()) {
                return {x, iteration + 1, residual_norm, false};
            }

            const auto beta = rz_new / rz_old;
            p = z + beta * p;
            rz_old = rz_new;
        }

        return {x, options.max_iterations, residual_norm, false};
    }

    template<typename MatrixType, typename VectorType, typename Preconditioner>
        requires detail::linear_system_type<MatrixType, VectorType> &&
                 detail::preconditioner_argument<Preconditioner>
    solver_result<VectorType> pcg(const MatrixType& A,
                                  const VectorType& b,
                                  Preconditioner preconditioner,
                                  solver_options options = {}) {
        return pcg(A, b, create_matrix<VectorType>(), preconditioner, options);
    }

    template<typename MatrixType, typename VectorType>
        requires detail::linear_system_type<MatrixType, VectorType>
    solver_result<VectorType> pcg(const MatrixType& A,
                                  const VectorType& b,
                                  VectorType x,
                                  solver_options options = {}) {
        return pcg(A, b, x, jacobi_preconditioner{}, options);
    }

    template<typename MatrixType, typename VectorType>
        requires detail::linear_system_type<MatrixType, VectorType>
    solver_result<VectorType> pcg(const MatrixType& A,
                                  const VectorType& b,
                                  solver_options options = {}) {
        return pcg(A, b, create_matrix<VectorType>(), options);
    }

    template<typename MatrixType, typename VectorType>
        requires detail::linear_system_type<MatrixType, VectorType>
    solver_result<VectorType> cg(const MatrixType& A,
                                 const VectorType& b,
                                 VectorType x,
                                 solver_options options = {}) {
        return pcg(A, b, x, identity_preconditioner{}, options);
    }

    template<typename MatrixType, typename VectorType>
        requires detail::linear_system_type<MatrixType, VectorType>
    solver_result<VectorType> cg(const MatrixType& A,
                                 const VectorType& b,
                                 solver_options options = {}) {
        return cg(A, b, create_matrix<VectorType>(), options);
    }
}
