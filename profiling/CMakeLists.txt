find_package(Vulkan)

file(GLOB_RECURSE HEADER_FILES ${CMAKE_CURRENT_LIST_DIR}/src/*.hpp)
file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

include_directories(${CMAKE_CURRENT_LIST_DIR}/src)

add_executable(profiling ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(profiling PRIVATE VulkanBase Vulkan::Vulkan ${CONAN_LIBS})