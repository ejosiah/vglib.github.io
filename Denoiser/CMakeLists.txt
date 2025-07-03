project(Denoiser VERSION 0.0.0)

find_package(OptiX REQUIRED)
_add_package_Cuda()

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

add_definitions(-DGLFW_INCLUDE_VULKAN)

file(GLOB_RECURSE CPP_FILES ${CMAKE_CURRENT_LIST_DIR}/source/*.cpp)
file(GLOB_RECURSE HPP_FILES ${CMAKE_CURRENT_LIST_DIR}/include/*.*)

add_library(Denoiser ${CPP_FILES} ${HPP_FILES})


target_include_directories(Denoiser PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

target_link_libraries(Denoiser PRIVATE VulkanBase ${LIBRARIES_DEBUG})