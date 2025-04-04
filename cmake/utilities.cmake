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


    if(EXISTS ${COMPILE_INCLUDE_DIRS})
        set(INCLUDE_DIRS ${COMPILE_INCLUDE_DIRS})
    endif()


    set(GLSL_SHADER_TYPE "")
    string(REGEX MATCH ".*\\.(rgen|rchit|rmiss).glsl" RAY_TRACE_STAGE ${COMPILE_SRC_FILE})
    if(RAY_TRACE_STAGE)
        set(GLSL_SHADER_TYPE "-fshader-stage=${CMAKE_MATCH_1}")
        string(REPLACE ".glsl" "" COMPILE_OUT_FILE ${COMPILE_OUT_FILE})
    endif()


    set(GLSLC_COMMAND "${GLSLC} -g -I ${INCLUDE_DIRS} ${GLSL_SHADER_TYPE} --target-spv=${COMPILE_SPV_VERSION} ${COMPILE_SRC_FILE} -o ${COMPILE_OUT_FILE}")
    execute_process(
        COMMAND ${GLSLC} -g -I ${INCLUDE_DIRS} ${GLSL_SHADER_TYPE} --target-spv=${COMPILE_SPV_VERSION} ${COMPILE_SRC_FILE} -o ${COMPILE_OUT_FILE}
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

    file(GLOB GLSL_SOURCE_FILES
        "${COMPILE_SRC_DIR}/*.vert"
        "${COMPILE_SRC_DIR}/*.frag"
        "${COMPILE_SRC_DIR}/*.geom"
        "${COMPILE_SRC_DIR}/*.comp"
        "${COMPILE_SRC_DIR}/*.tese"
        "${COMPILE_SRC_DIR}/*.tesc"
        "${COMPILE_SRC_DIR}/*.rgen*"
        "${COMPILE_SRC_DIR}/*.rmiss*"
        "${COMPILE_SRC_DIR}/*.rchit*"
        "${COMPILE_SRC_DIR}/*.rahit*"
        "${COMPILE_SRC_DIR}/*.rint*"
        "${COMPILE_SRC_DIR}/*.rcall*"
        "${COMPILE_SRC_DIR}/*.mesh*"
        "${COMPILE_SRC_DIR}/*.msh*"
        "${COMPILE_SRC_DIR}/*.task*"
        "${COMPILE_SRC_DIR}/*.tsk*"
    )

    file(MAKE_DIRECTORY ${COMPILE_OUT_DIR})

    foreach(SHADER_SOURCE IN ITEMS ${GLSL_SOURCE_FILES})
        get_filename_component(SHADER_FILE_NAME ${SHADER_SOURCE} NAME)
        set(SPV_FILE "${COMPILE_OUT_DIR}/${SHADER_FILE_NAME}.spv")
        compile_glsl(SRC_FILE ${SHADER_SOURCE} OUT_FILE ${SPV_FILE} INCLUDE_DIRS ${COMPILE_INCLUDE_DIRS})
    endforeach()

endfunction()