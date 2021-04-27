# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(Segmentation)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../out")
set(LIBRARY_OUTPUT_PATH "../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")
set(CMAKE_SKIP_RPATH TRUE)

set(INC_PATH $ENV{DDK_PATH})

if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

set(LIB_PATH $ENV{NPU_HOST_LIB})

if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/acllib/lib64/stub/")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()


find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Header path
include_directories(
    ${INC_PATH}/acllib/include/
)


# add host lib path
link_directories(
    /usr/local/lib
    ${LIB_PATH}
    ${INC_PATH}/atc/lib64
)

aux_source_directory(. LIB_SRC)
list(REMOVE_ITEM LIB_SRC "./example.cpp")

add_library(segmentation SHARED ${LIB_SRC})
target_link_libraries(segmentation ascendcl stdc++ ${OpenCV_LIBS})

add_executable(main ./example.cpp)
target_link_libraries(main segmentation ${OpenCV_LIBS})
