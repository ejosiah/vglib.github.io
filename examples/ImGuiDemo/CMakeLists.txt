
include_directories(include)

file(GLOB_RECURSE CPP_FILES ${CMAKE_CURRENT_LIST_DIR}/source/*.cpp)
file(GLOB_RECURSE HPP_FILES ${CMAKE_CURRENT_LIST_DIR}/include/*.*)

add_executable(ImGuiDemo ${CPP_FILES} ${HPP_FILES})
target_link_libraries(ImGuiDemo VulkanBase ImGuiPlugin)