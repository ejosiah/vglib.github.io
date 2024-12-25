file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/*.hpp ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

add_executable(scripts ${SRC_FILES})