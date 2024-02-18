message("cmake build type is ${CMAKE_BUILD_TYPE} .")
list(APPEND tpu_asr_model_srcs ./tpu_asr_model.cc)
message(STATUS "Use src_files: [ ${tpu_asr_model_srcs} ] to compile tpu_asr_model .")
add_definitions(-DUSE_TPU)
include(${CMAKE_SOURCE_DIR}/cmake/tpu.cmake)

SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        message( "${CMAKE_SYSTEM_PROCESSOR} mode, starting......")
        check_and_replace_path(SDK)
        set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
        set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
        set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
        set(BM_LIBS bmlib bmrt bmcv yuv)
        include_directories("../../../../include/")
        include_directories("${SDK}/include/")
        link_directories("${SDK}/lib/")
        message("SDK: " ${SDK})

        # compile tpu_asr_model
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)
        add_library(tpu_asr_model STATIC ${tpu_asr_model_srcs})
        target_link_libraries(tpu_asr_model PUBLIC ${BM_LIBS} -lpthread)
    else()
        message( "${CMAKE_SYSTEM_PROCESSOR} mode, starting......")
        # use libbmrt libbmlib
        find_package(libsophon REQUIRED)
        include_directories(${LIBSOPHON_INCLUDE_DIRS})
        link_directories(${LIBSOPHON_LIB_DIRS})
        message("libsophon dir: " ${LIBSOPHON_INCLUDE_DIRS})

        # compile tpu_asr_model
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)
        add_library(tpu_asr_model STATIC ${tpu_asr_model_srcs})
        target_link_libraries(tpu_asr_model PUBLIC ${the_libbmlib.so} ${the_libbmrt.so} -lpthread)
    endif()
else()
    message(FATAL_ERROR "Unsupported CMake System Name '${CMAKE_SYSTEM_NAME}' (expected 'Linux')")
endif()



