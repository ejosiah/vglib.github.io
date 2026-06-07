#pragma once

#include <fmt/format.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace matrix_market {
    struct entry {
        uint32_t rowIndex;
        uint32_t colIndex;
        float value;
    };

    struct matrix {
        std::vector<entry> data;
        uint32_t rowCount{};
        uint32_t colCount{};

        [[nodiscard]] auto begin() {
            return data.begin();
        }

        [[nodiscard]] auto end() {
            return data.end();
        }

        [[nodiscard]] auto begin() const {
            return data.begin();
        }

        [[nodiscard]] auto end() const {
            return data.end();
        }

        [[nodiscard]] auto cbegin() const noexcept {
            return data.cbegin();
        }

        [[nodiscard]] auto cend() const noexcept {
            return data.cend();
        }

        float operator()(size_t r, size_t c) const {
            auto itr = std::find_if(cbegin(), cend(), [&](const auto& e) {
                return e.rowIndex == r && e.colIndex == c;
            });
            assert(itr != cend());
            return itr->value;
        }

        std::vector<float> row(size_t r) const {
            std::vector<float> result;

            for (auto [row, _, value] : data) {
                if (row == r) {
                    result.push_back(value);
                }
            }

            return result;
        }
    };

    [[nodiscard]]
    inline matrix load(const std::filesystem::path& path) {
        std::ifstream in{path.string()};

        if (!in.is_open()) {
            throw std::runtime_error{fmt::format("unable to open path: {}", path.string())};
        }

        while (true) {
            if (in.peek() != '%') {
                break;
            }

            in.ignore(1024, '\n');
        }

        uint32_t rowIndex{};
        uint32_t colIndex{};
        uint32_t nonZeroCount{};
        in >> rowIndex >> colIndex >> nonZeroCount;

        matrix m{.rowCount = rowIndex, .colCount = colIndex};
        float value{};

        for (uint32_t i = 0; i < nonZeroCount; ++i) {
            in >> rowIndex >> colIndex >> value;
            m.data.emplace_back(rowIndex - 1, colIndex - 1, value);
        }

        return m;
    }
}
