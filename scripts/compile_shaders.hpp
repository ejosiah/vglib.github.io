#pragma once

#include <cinttypes>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <format>
#include <map>
#include <span>
#include <functional>
#include <regex>
#include <optional>
#include <sstream>
#include <tuple>

namespace compile_shaders_internal {
    namespace fs = std::filesystem;

    auto path_to_key(const fs::path& path) {
        auto key = std::regex_replace(path.string(), std::regex("C:\\\\"), "");
        key = std::regex_replace(key, std::regex("\\."), "_");  // TODO determine path based on os i.e windows is \ and linux is /
        key = std::regex_replace(key, std::regex("\\\\"), "_");
        return key;
    }

    auto remove_prefix(std::string key, std::optional<std::filesystem::path> prefix_to_strip) {
        if(prefix_to_strip.has_value()) {
            auto prefix = path_to_key(*prefix_to_strip);
            key = std::regex_replace(key, std::regex(prefix), "");
            key = std::regex_replace(key, std::regex("^_"), "");
            key = std::regex_replace(key, std::regex("_spv"), "");
        }
        return key;
    }

    void process(const fs::path& current_path,
                 std::map<std::string, std::vector<std::string>>& shader_map,
                 const std::optional<std::filesystem::path>& prefix_to_strip) {
        if(fs::is_directory(current_path)) {
            for ( const auto& entry : fs::directory_iterator(current_path)) {
                process(entry.path(), shader_map, prefix_to_strip);
            }
        } else if(current_path.extension() == ".spv") {

            auto fin = std::ifstream{current_path.string(), std::ios::binary | std::ios::ate};
            if (!fin.good()) {
                std::cout << std::format("unable to open {} for reading", current_path.string());
                std::terminate();
            }
            const auto size = fin.tellg();
            std::vector<char> buf(size);

            fin.seekg(std::ios::beg);
            fin.read(buf.data(), size);

            std::span<uint32_t> source_code{reinterpret_cast<uint32_t *>(buf.data()), size / sizeof(uint32_t)};

            std::vector<std::string> source_code_hex;
            for (auto x: source_code) {
                source_code_hex.emplace_back(std::format("{:#010x}", x));
            }
            auto key = remove_prefix(path_to_key(current_path), prefix_to_strip);
            shader_map.insert(std::make_pair(key, source_code_hex));
        }
    }

    std::string extract_source(const std::vector<std::string>& source_code) {
        auto source = std::stringstream{};
        for(int i = 0; i < source_code.size(); i += 8) {
            for(auto j = 0; j < 8; ++j) {
                auto index = i + j;
                if(index < source_code.size()) {
                    source << source_code[index] << ",";
                }
            }
            source << "\n\t";
        }
        return source.str();
    }

    std::tuple<std::string, std::string> generate_output(const std::map<std::string, std::vector<std::string>>& shader_map, std::string_view output_filename) {
        auto header_output = std::stringstream{};
        auto source_output = std::stringstream {};

        header_output << "#pragma once\n";
        header_output << "#include <vector>\n\n";
        source_output << "#include \"" << output_filename << ".hpp\"\n\n";

        for(const auto& [key, source] : shader_map) {
            header_output << "extern std::vector<uint32_t> " << key << ";\n";
            source_output <<  "std::vector<uint32_t> "<< key << "{\n\t" << extract_source(source) << "\n};\n\n";
        }

        return std::make_tuple(header_output.str(), source_output.str());
    }

    void write_to_disk(const std::string& source, std::string_view path) {
        auto fout = std::ofstream { std::string{path} };
        if(!fout.good()) {
            std::cout << std::format("unable to write {} to disk", path);
        }

        fout.write(source.data(), source.size());
        std::cout << path << " successfully written to disk\n";
    }

    void write_output(std::tuple<std::string, std::string> output_files, std::string_view filename) {
        auto [header, source] = output_files;
        write_to_disk(header, std::format("{}.hpp", filename));
        write_to_disk(source, std::format("{}.cpp", filename));
    }
}

inline std::string compile_shader_usage();

inline void compile_shaders(const std::filesystem::path& src_folder,
                            std::string_view output_filename,
                            std::optional<std::filesystem::path> prefix_to_strip = {}) {
    using namespace compile_shaders_internal;

    std::map<std::string, std::vector<std::string>> shader_map;
    process(src_folder, shader_map, prefix_to_strip);
    write_output(generate_output(shader_map, output_filename), output_filename);
}