include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

add_definitions(-DGLFW_INCLUDE_VULKAN)

file(GLOB_RECURSE CPP_FILES ${CMAKE_CURRENT_LIST_DIR}/source/*.cpp)
file(GLOB_RECURSE HPP_FILES ${CMAKE_CURRENT_LIST_DIR}/include/*.*)

add_library(BulletPhysicsPlugin ${CPP_FILES} ${HPP_FILES})
#add_library(VgLib::BulletPhysicsPlugin ALIAS BulletPhysicsPlugin)

target_include_directories(BulletPhysicsPlugin PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

target_link_libraries(BulletPhysicsPlugin PRIVATE VulkanBase)