project(ImGuiPlugin VERSION 0.0.0)

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

add_definitions(-DGLFW_INCLUDE_VULKAN)

file(GLOB_RECURSE CPP_FILES ${CMAKE_CURRENT_LIST_DIR}/source/*.cpp)
file(GLOB_RECURSE HPP_FILES ${CMAKE_CURRENT_LIST_DIR}/include/*.*)

add_library(ImGuiPlugin ${CPP_FILES} ${HPP_FILES})
#add_library(VgLib::ImGuiPlugin ALIAS ImGuiPlugin)


target_include_directories(ImGuiPlugin PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

target_link_libraries(ImGuiPlugin PRIVATE VulkanBase)