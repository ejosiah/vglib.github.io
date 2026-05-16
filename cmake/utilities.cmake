include_guard(GLOBAL)

function(_attach_glsl_shader_target SHADER_TARGET SOURCE_DIR)
    if(NOT TARGET ${SHADER_TARGET})
        return()
    endif()

    get_property(_DIRECTORY_TARGETS DIRECTORY "${SOURCE_DIR}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_TARGET IN LISTS _DIRECTORY_TARGETS)
        if(_TARGET STREQUAL "${SHADER_TARGET}")
            continue()
        endif()

        get_target_property(_TARGET_TYPE ${_TARGET} TYPE)
        if(_TARGET_TYPE STREQUAL "EXECUTABLE"
                OR _TARGET_TYPE STREQUAL "STATIC_LIBRARY"
                OR _TARGET_TYPE STREQUAL "SHARED_LIBRARY"
                OR _TARGET_TYPE STREQUAL "MODULE_LIBRARY"
                OR _TARGET_TYPE STREQUAL "OBJECT_LIBRARY")
            add_dependencies(${_TARGET} ${SHADER_TARGET})
        endif()
    endforeach()
endfunction()

function(compile_glsl)
    if(NOT DEFINED GLSLC)
        message(FATAL_ERROR "glsl complier path not defined")
    endif()
    set(noValues "")
    set(singleValues SRC_FILE OUT_FILE SPV_VERSION)
    set(multiValues INCLUDE_DIRS DEPENDS)

    cmake_parse_arguments(COMPILE "${noValues}" "${singleValues}" "${multiValues}" ${ARGN})

    if(NOT COMPILE_SPV_VERSION)
        set(COMPILE_SPV_VERSION spv1.6)
    endif()

    if(NOT EXISTS ${COMPILE_SRC_FILE})
        message(FATAL_ERROR "${COMPILE_SRC_FILE} not found")
    endif()


    if(COMPILE_INCLUDE_DIRS)
        set(INCLUDE_DIRS ${COMPILE_INCLUDE_DIRS})
    endif()

    set(GLSLC_INCLUDES "")
    foreach(dir IN LISTS INCLUDE_DIRS)
        list(APPEND GLSLC_INCLUDES -I "${dir}")
    endforeach()

    set(GLSL_SHADER_TYPE "")
    string(REGEX MATCH ".*\\.(rgen|rchit|rmiss).glsl" RAY_TRACE_STAGE ${COMPILE_SRC_FILE})
    if(RAY_TRACE_STAGE)
        set(GLSL_SHADER_TYPE "-fshader-stage=${CMAKE_MATCH_1}")
        string(REPLACE ".glsl" "" COMPILE_OUT_FILE ${COMPILE_OUT_FILE})
    endif()


    get_filename_component(SHADER_SRC_FILE ${COMPILE_SRC_FILE} NAME)
    get_filename_component(SHADER_OUT_DIR ${COMPILE_OUT_FILE} DIRECTORY)

    set(GLSLC_COMMAND "${GLSLC}" -g)
    list(APPEND GLSLC_COMMAND ${GLSLC_INCLUDES})
    if(GLSL_SHADER_TYPE)
        list(APPEND GLSLC_COMMAND "${GLSL_SHADER_TYPE}")
    endif()
    list(APPEND GLSLC_COMMAND --target-spv=${COMPILE_SPV_VERSION} "${COMPILE_SRC_FILE}" -o "${COMPILE_OUT_FILE}")

    add_custom_command(
        OUTPUT "${COMPILE_OUT_FILE}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUT_DIR}"
        COMMAND ${GLSLC_COMMAND}
        DEPENDS "${COMPILE_SRC_FILE}" ${COMPILE_DEPENDS}
        COMMENT "Compiling GLSL shader ${SHADER_SRC_FILE}"
        VERBATIM
    )


endfunction()

function(compile_glsl_directory)
    set(noValues "")
    set(singleValues SRC_DIR OUT_DIR)
    set(multiValues INCLUDE_DIRS)

    cmake_parse_arguments(COMPILE "${noValues}" "${singleValues}" "${multiValues}" ${ARGN})

    # Normalize to absolute paths (avoids RELATIVE_PATH error)
    get_filename_component(_SRC_ROOT "${COMPILE_SRC_DIR}" ABSOLUTE)
    get_filename_component(_OUT_ROOT "${COMPILE_OUT_DIR}" ABSOLUTE)

    file(GLOB_RECURSE GLSL_SOURCE_FILES CONFIGURE_DEPENDS
            "${_SRC_ROOT}/*.vert"
            "${_SRC_ROOT}/*.frag"
            "${_SRC_ROOT}/*.geom"
            "${_SRC_ROOT}/*.comp"
            "${_SRC_ROOT}/*.tese"
            "${_SRC_ROOT}/*.tesc"
            "${_SRC_ROOT}/*.rgen*"
            "${_SRC_ROOT}/*.rmiss*"
            "${_SRC_ROOT}/*.rchit*"
            "${_SRC_ROOT}/*.rahit*"
            "${_SRC_ROOT}/*.rint*"
            "${_SRC_ROOT}/*.rcall*"
            "${_SRC_ROOT}/*.mesh*"
            "${_SRC_ROOT}/*.msh*"
            "${_SRC_ROOT}/*.task*"
            "${_SRC_ROOT}/*.tsk*"
    )

    file(GLOB_RECURSE GLSL_DEPENDENCY_FILES CONFIGURE_DEPENDS
            "${_SRC_ROOT}/*.glsl"
    )

    foreach(_INCLUDE_DIR IN LISTS COMPILE_INCLUDE_DIRS)
        get_filename_component(_INCLUDE_ROOT "${_INCLUDE_DIR}" ABSOLUTE)
        if(EXISTS "${_INCLUDE_ROOT}")
            file(GLOB_RECURSE _INCLUDE_DEPENDENCY_FILES CONFIGURE_DEPENDS
                    "${_INCLUDE_ROOT}/*.glsl"
            )
            list(APPEND GLSL_DEPENDENCY_FILES ${_INCLUDE_DEPENDENCY_FILES})
        endif()
    endforeach()

    if(GLSL_DEPENDENCY_FILES)
        list(REMOVE_DUPLICATES GLSL_DEPENDENCY_FILES)
    endif()

    set(SPV_SOURCE_FILES "")
    foreach(SHADER_SOURCE IN LISTS GLSL_SOURCE_FILES)
        get_filename_component(_SHADER_FILE "${SHADER_SOURCE}" NAME)
        get_filename_component(_SHADER_DIR  "${SHADER_SOURCE}" DIRECTORY)

        # Relative dir from the root of the shader source tree
        file(RELATIVE_PATH _RELDIR "${_SRC_ROOT}" "${_SHADER_DIR}")

        # Build snake_case prefix from parent dirs (excluding _SRC_ROOT)
        set(_PREFIX "")
        if (NOT _RELDIR STREQUAL "." AND NOT _RELDIR STREQUAL "")
            string(REPLACE "\\" "/" _PREFIX "${_RELDIR}")                      # normalize slashes
            string(REGEX REPLACE "([a-z0-9])([A-Z])" "\\1_\\2" _PREFIX "${_PREFIX}" ) # camel->snake
            string(REGEX REPLACE "[^A-Za-z0-9]+" "_" _PREFIX "${_PREFIX}")     # non-alnum -> _
            string(REGEX REPLACE "_+" "_" _PREFIX "${_PREFIX}")                 # collapse __
            string(REGEX REPLACE "^_|_$" "" _PREFIX "${_PREFIX}")              # trim edges
            string(TOLOWER "${_PREFIX}" _PREFIX)                                # lower-case
            set(_PREFIX "${_PREFIX}_")
        endif()

        set(SPV_FILE "${_OUT_ROOT}/${_PREFIX}${_SHADER_FILE}.spv")
        string(REGEX MATCH ".*\\.(rgen|rchit|rmiss).glsl" _RAY_TRACE_STAGE "${SHADER_SOURCE}")
        if(_RAY_TRACE_STAGE)
            string(REPLACE ".glsl" "" SPV_FILE "${SPV_FILE}")
        endif()

        compile_glsl(SRC_FILE "${SHADER_SOURCE}" OUT_FILE "${SPV_FILE}" INCLUDE_DIRS ${COMPILE_INCLUDE_DIRS} DEPENDS ${GLSL_DEPENDENCY_FILES})
        list(APPEND SPV_SOURCE_FILES "${SPV_FILE}")
    endforeach()

    if(SPV_SOURCE_FILES)
        file(RELATIVE_PATH _SHADER_TARGET_SUFFIX "${CMAKE_SOURCE_DIR}" "${_SRC_ROOT}")
        string(REPLACE "\\" "/" _SHADER_TARGET_SUFFIX "${_SHADER_TARGET_SUFFIX}")
        string(REGEX REPLACE "[^A-Za-z0-9]+" "_" _SHADER_TARGET_SUFFIX "${_SHADER_TARGET_SUFFIX}")
        string(REGEX REPLACE "_+" "_" _SHADER_TARGET_SUFFIX "${_SHADER_TARGET_SUFFIX}")
        string(REGEX REPLACE "^_|_$" "" _SHADER_TARGET_SUFFIX "${_SHADER_TARGET_SUFFIX}")
        string(TOLOWER "${_SHADER_TARGET_SUFFIX}" _SHADER_TARGET_SUFFIX)
        if(_SHADER_TARGET_SUFFIX MATCHES "(^|_)shaders$")
            set(_SHADER_TARGET "${_SHADER_TARGET_SUFFIX}")
        else()
            set(_SHADER_TARGET "${_SHADER_TARGET_SUFFIX}_shaders")
        endif()

        add_custom_target(${_SHADER_TARGET} ALL
            DEPENDS ${SPV_SOURCE_FILES}
            SOURCES ${GLSL_SOURCE_FILES} ${GLSL_DEPENDENCY_FILES}
        )
        cmake_language(EVAL CODE
            "cmake_language(DEFER DIRECTORY \"${CMAKE_CURRENT_SOURCE_DIR}\" CALL _attach_glsl_shader_target \"${_SHADER_TARGET}\" \"${CMAKE_CURRENT_SOURCE_DIR}\")"
        )
    endif()
endfunction()
