#include "compile_shaders.hpp"

int main(int argc, char** argv) {
    if(argc == 1) {
        std::cout << "script to run required\n";
        std::exit(3);
    }
    compile_shader_params cParams{};
    cParams.src_folder = argv[1];
    cParams.output_header_path = argv[2];
    cParams.output_src_path = argv[3];
    cParams.output_filename = argv[4];
    cParams.prefix_to_strip = argc >= 6 ? std::optional<std::string>{argv[5]} : std::nullopt;
    compile_shaders(cParams);
}