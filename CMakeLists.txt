cmake_minimum_required(VERSION 3.19)
project(vglib VERSION 0.0.0)
enable_testing()

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake ${CMAKE_MODULE_PATH})

find_program(GLSLC glslc REQUIRED)

if(CMAKE_GENERATOR STREQUAL "Xcode")
    add_compile_options(-Wno-nullability-completeness)
endif()

message(STATUS "setting up conan")
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

include(cmake/utilities.cmake)
include(cmake/setup.cmake)

find_package(OptiX REQUIRED)
_add_package_Cuda()

add_definitions(
        -DGLM_FORCE_RADIANS
        -DGLFW_INCLUDE_VULKAN
        -DGLM_FORCE_SWIZZLE
)


include_directories(${OptiX_INCLUDE})
include_directories(3rdParty/include)


message(STATUS "optix include directory: ${OptiX_INCLUDE}")

include(GNUInstallDirs)

add_subdirectory(VulkanBase)
add_subdirectory(ImGuiPlugin)
add_subdirectory(BulletPhysicsPlugin)
#add_subdirectory(camera)

install(TARGETS VulkanBase
        EXPORT VgLib
        CONFIGURATIONS Debug
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/Debug
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/Debug
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/Debug
#        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)


install(TARGETS VulkanBase
        CONFIGURATIONS Release RelWithDebInfo MinSizeRel
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/Release
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/Release
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/Release
#        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${VGLIB_INCLUDE_DIRS} DESTINATION include FILES_MATCHING PATTERN *.h)
install(DIRECTORY ${VGLIB_INCLUDE_DIRS} DESTINATION include FILES_MATCHING PATTERN *.hpp)

file(GLOB_RECURSE BASE_HPP_FILES VulkanBase/include/*.*)
#install(FILES ${BASE_HPP_FILES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY VulkanBase/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vgl)

install(EXPORT VgLib
        FILE VgLigConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VgLib
        NAMESPACE VgLib::
 )


set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
include(InstallRequiredSystemLibraries)
