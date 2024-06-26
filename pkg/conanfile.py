from conans import ConanFile, CMake, tools


class VgLibConan(ConanFile):
    name = "vglib"
    version = "0.3.0"
    license = "<Put the package license here>"
    author = "Josiah Ebhomenye joebhomenye@gmail.com"
    url = "<Package recipe repository url here, for issues about the package>"
    description = "vulkan utility library"
    topics = ("vulkan", "graphics", "3d")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}
    generators = "cmake"
    requires = ["assimp/5.2.2",
                "glm/0.9.9.8",
                "glfw/3.3.2",
                "stb/20200203",
                "spdlog/1.8.2",
                "freetype/2.12.1",
                "imgui/1.82",
                "boost/1.80.0",
                "gtest/1.11.0",
                "argparse/2.1",
                "bullet3/3.17",
                "entt/3.8.1",
                "vhacd/0.1",
                "meshoptimizer/0.17",
                "openexr/3.1.5",
                "taskflow/3.4.0",
                "openvdb/8.0.1"
                ]

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def source(self):
        self.run("git clone git@github.com:ejosiah/vglib.github.io.git vglib")
        self.run("git -C \"%s/vglib\" checkout v%s" % (self.source_folder, self.version))

    def build(self):
        self.run('conan install \"%s/vglib\" -s build_type=%s' % (self.source_folder, self.settings.build_type.value))
        cmake = CMake(self)
        cmake.configure(source_folder="vglib")
        # cmake.build()

        # Explicit way:
        self.run('cmake \"%s/vglib\" %s'
                 % (self.source_folder, cmake.command_line))
        self.run("cmake \"%s/vglib\" -D BUILD_EXAMPLES:BOOL=OFF" % self.source_folder)
        self.run("cmake --build . %s" % cmake.build_config)

    def package(self):
        self.copy("VulkanBase/*.h*", dst="include/vglib", src="vglib", keep_path=False)
        self.copy("3rdParty/*.h*", dst="include/vglib", src="vglib", keep_path=False)
        self.copy("ImGuiPlugin/*.h*", dst="include/vglib", src="vglib", keep_path=False)
        self.copy("*.lib", dst="lib", keep_path=False)
        self.copy("*.dll", dst="bin", keep_path=False)
        self.copy("*.so", dst="lib", keep_path=False)
        self.copy("*.dylib", dst="lib", keep_path=False)
        self.copy("*.a", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["VulkanBase", "ImGuiPlugin"]

