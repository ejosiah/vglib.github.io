#include "compile_shaders.hpp"

int main(int argc, char** argv) {
    if(argc == 1) {
        std::cout << "script to run required\n";
        std::exit(3);
    }
    auto shader_path = argv[1];
    auto output_filename = argv[2];
    std::optional<std::string> prefix_to_strip = argc >= 4 ? std::optional<std::string>{argv[3]} : std::nullopt;
    compile_shaders(shader_path, output_filename, prefix_to_strip);
}