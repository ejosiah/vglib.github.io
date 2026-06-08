project(VulkanBase VERSION 0.0.0)

find_package(Vulkan)
find_package(assimp)
find_package(glm)
find_package(glfw3)
find_package(stb)
find_package(spdlog)
find_package(freetype)
find_package(imgui)
find_package(GTest)
find_package(argparse)
find_package(Bullet)
find_package(EnTT)
find_package(meshoptimizer)
find_package(OpenEXR)
find_package(Taskflow)
find_package(ZLIB)
find_package(Ktx)
find_package(nlohmann_json)



if (NOT NO_OPENCL)
    find_package(OpenCL QUIET)
endif()

add_definitions(
        -DGLM_FORCE_RADIANS
        -DGLFW_INCLUDE_VULKAN
        -DGLM_FORCE_SWIZZLE
)

set(CONAN_LIBS_DEP  assimp::assimp glm::glm glfw stb::stb spdlog::spdlog imgui::imgui
                    gtest::gtest argparse::argparse Bullet::Bullet EnTT::EnTT meshoptimizer::meshoptimizer
                    openexr::openexr Taskflow::Taskflow ZLIB::ZLIB Freetype::Freetype KTX::ktx nlohmann_json::nlohmann_json)


set(VGLIB_INCLUDE_DIRS
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/utility>
)

set(VGLIB_INSTALL_INCLUDE_DIRS
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/utility>
        )

file(GLOB_RECURSE CPP_FILES ${CMAKE_CURRENT_LIST_DIR}/source/*.cpp)
file(GLOB_RECURSE HPP_FILES ${CMAKE_CURRENT_LIST_DIR}/include/*.*)
file(GLOB_RECURSE TEST_FILES ${CMAKE_CURRENT_LIST_DIR}/tests/*.*)

add_library(VulkanBase ${CPP_FILES} ${HPP_FILES})
add_library(VgLib::VulkanBase ALIAS VulkanBase)
target_link_libraries(VulkanBase PUBLIC Vulkan::Vulkan ${CONAN_LIBS_DEP})

target_include_directories(VulkanBase PUBLIC ${VGLIB_INCLUDE_DIRS} ${VGLIB_INSTALL_INCLUDE_DIRS})

if (OpenCL_FOUND)
    include_directories("${OpenCL_INCLUDE_DIRS}")
    add_definitions( -DOPENCL_FOUND=1 )
    target_link_libraries(VulkanBase PUBLIC ${OpenCL_LIBRARIES})
    message(STATUS "openCL found ${OpenCL_LIBRARIES}")
#    add_compile_definitions(OPENCL_AVAILABLE)
endif()

add_executable(VulkanBaseTest  ${TEST_FILES})
set(LINEAR_SOLVER_JACOBI_SPV ${CMAKE_CURRENT_BINARY_DIR}/test_shaders/linalg/jacobi.comp.spv)
add_custom_command(
        OUTPUT ${LINEAR_SOLVER_JACOBI_SPV}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/test_shaders/linalg
        COMMAND ${GLSLC} -g -fshader-stage=compute --target-spv=spv1.6
                ${CMAKE_CURRENT_LIST_DIR}/../data/shaders/linalg/jacobi.comp
                -o ${LINEAR_SOLVER_JACOBI_SPV}
        DEPENDS ${CMAKE_CURRENT_LIST_DIR}/../data/shaders/linalg/jacobi.comp
        VERBATIM)
add_custom_target(VulkanBaseTestShaders DEPENDS ${LINEAR_SOLVER_JACOBI_SPV})
add_dependencies(VulkanBaseTest VulkanBaseTestShaders)
file(TO_CMAKE_PATH "${LINEAR_SOLVER_JACOBI_SPV}" LINEAR_SOLVER_JACOBI_SPV_DEFINE)
target_compile_definitions(VulkanBaseTest PRIVATE LINEAR_SOLVER_JACOBI_SPV="${LINEAR_SOLVER_JACOBI_SPV_DEFINE}")
target_link_libraries(VulkanBaseTest PRIVATE VulkanBase Vulkan::Vulkan  gtest::gtest nlohmann_json::nlohmann_json)

include(GoogleTest)
gtest_discover_tests(VulkanBaseTest)
