include_guard(GLOBAL)

function(compile_glsl)
    if(NOT DEFINED GLSLC)
        message(FATAL_ERROR "glsl complier path not defined")
    endif()
    set(noValues "")
    set(singleValues SRC_FILE OUT_FILE SPV_VERSION)
    set(multiValues INCLUDE_DIRS)

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


    set(GLSLC_COMMAND "${GLSLC} -g ${GLSLC_INCLUDES} ${GLSL_SHADER_TYPE} --target-spv=${COMPILE_SPV_VERSION} ${COMPILE_SRC_FILE} -o ${COMPILE_OUT_FILE}")
    execute_process(
        COMMAND ${GLSLC} -g ${GLSLC_INCLUDES} ${GLSL_SHADER_TYPE} --target-spv=${COMPILE_SPV_VERSION} ${COMPILE_SRC_FILE} -o ${COMPILE_OUT_FILE}
        RESULT_VARIABLE GLSLC_COMPILE_OUTPUT
    )
    get_filename_component(SHADER_SRC_FILE ${COMPILE_SRC_FILE} NAME)
    if(${GLSLC_COMPILE_OUTPUT})
        message(STATUS ${GLSLC_COMMAND})
        message(FATAL_ERROR "compile failed for ${SHADER_SRC_FILE}, reason: ${GLSLC_COMPILE_OUTPUT}")
    endif()


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

    file(MAKE_DIRECTORY "${_OUT_ROOT}")

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
        compile_glsl(SRC_FILE "${SHADER_SOURCE}" OUT_FILE "${SPV_FILE}" INCLUDE_DIRS ${COMPILE_INCLUDE_DIRS})
    endforeach()
endfunction()