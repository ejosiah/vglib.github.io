#####################################################################################
# Optional CUDA package
# see https://cmake.org/cmake/help/v3.3/module/FindCUDA.html
#
macro(_add_package_Cuda)
    if(CUDA_TOOLKIT_ROOT_DIR)
        string(REPLACE "\\" "/" CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR})
    endif()
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        add_definitions("-DCUDA_PATH=R\"(${CUDA_TOOLKIT_ROOT_DIR})\"")
        Message(STATUS "--> using package CUDA (${CUDA_VERSION})")
        add_definitions(-DNVP_SUPPORTS_CUDA)
        include_directories(${CUDA_INCLUDE_DIRS})
        LIST(APPEND LIBRARIES_OPTIMIZED ${CUDA_LIBRARIES} )
        LIST(APPEND LIBRARIES_DEBUG ${CUDA_LIBRARIES} )
        # STRANGE: default CUDA package finder from cmake doesn't give anything to find cuda.lib
        if(WIN32)
            if((ARCH STREQUAL "x86"))
                LIST(APPEND LIBRARIES_DEBUG "${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32/cuda.lib" )
                LIST(APPEND LIBRARIES_DEBUG "${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32/cudart.lib" )
                LIST(APPEND LIBRARIES_OPTIMIZED "${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32/cuda.lib" )
                LIST(APPEND LIBRARIES_OPTIMIZED "${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32/cudart.lib" )
            else()
                LIST(APPEND LIBRARIES_DEBUG "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib" )
                LIST(APPEND LIBRARIES_DEBUG "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart.lib" )
                LIST(APPEND LIBRARIES_DEBUG "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nvrtc.lib" )
                LIST(APPEND LIBRARIES_OPTIMIZED "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib" )
                LIST(APPEND LIBRARIES_OPTIMIZED "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart.lib" )
                LIST(APPEND LIBRARIES_OPTIMIZED "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nvrtc.lib" )
            endif()
        else()
            LIST(APPEND LIBRARIES_DEBUG "libcuda.so" )
            LIST(APPEND LIBRARIES_DEBUG "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so" )
            LIST(APPEND LIBRARIES_DEBUG "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvrtc.so" )
            LIST(APPEND LIBRARIES_OPTIMIZED "libcuda.so" )
            LIST(APPEND LIBRARIES_OPTIMIZED "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so" )
            LIST(APPEND LIBRARIES_OPTIMIZED "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvrtc.so" )
        endif()
        #LIST(APPEND PACKAGE_SOURCE_FILES ${CUDA_HEADERS} ) Not available anymore with cmake 3.3... we might have to list them by hand
        # source_group(CUDA FILES ${CUDA_HEADERS} )  Not available anymore with cmake 3.3
    else()
        Message(STATUS "--> NOT using package CUDA")
    endif()
endmacro()
# this macro is needed for the samples to add this package, although not needed
# this happens when the nvpro_core library was built with these stuff in it
# so many samples can share the same library for many purposes
macro(_optional_package_Cuda)
    if(CUDA_FOUND)
        _add_package_Cuda()
    endif(CUDA_FOUND)
endmacro(_optional_package_Cuda)