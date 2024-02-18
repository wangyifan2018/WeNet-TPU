#Usage: check_and_replace_path(SDK)
#Input src path, output absolute path.
function(check_and_replace_path ARG_NAME)
    if(IS_ABSOLUTE ${${ARG_NAME}})
        return()
    endif()
    set(PATH_TO_CHECK ${CMAKE_CURRENT_BINARY_DIR}/${${ARG_NAME}})
    if(EXISTS ${PATH_TO_CHECK})
        message("Path ${PATH_TO_CHECK} exists")
        get_filename_component(ABSOLUTE_PATH ${PATH_TO_CHECK} ABSOLUTE)
        if(EXISTS ${ABSOLUTE_PATH})
            set(${ARG_NAME} ${ABSOLUTE_PATH} PARENT_SCOPE)
        else()
            message(FATAL_ERROR "Invalid path!")
        endif()
    else()
        message(FATAL_ERROR "Path ${PATH_TO_CHECK} does not exist")
    endif()
endfunction()

if(TPU)
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
    else()
      message( "${CMAKE_SYSTEM_PROCESSOR} mode, starting......")
      # use libbmrt libbmlib
      find_package(libsophon REQUIRED)
      include_directories(${LIBSOPHON_INCLUDE_DIRS})
      link_directories(${LIBSOPHON_LIB_DIRS})
      message("libsophon dir: " ${LIBSOPHON_INCLUDE_DIRS})
    endif()
  else()
    message(FATAL_ERROR "Unsupported CMake System Name '${CMAKE_SYSTEM_NAME}' (expected 'Linux')")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=gold")
endif()