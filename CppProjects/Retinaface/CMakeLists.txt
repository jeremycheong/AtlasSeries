cmake_minimum_required (VERSION 2.8.12)

project(Retinaface)

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../out")
set(LIBRARY_OUTPUT_PATH "../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")


# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self")
if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmaybe-uninitialized")
endif()

### to use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
if (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

### ascend libs
add_definitions(-DENABLE_DVPP_INTERFACE)
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

# Header path
include_directories(
    ${INC_PATH}/acllib/include/
    ${PROJECT_SOURCE_DIR}/include   # 3rd header path
)

# add host lib path
link_directories(
    ${LIB_PATH}
    ${INC_PATH}/atc/lib64
    ${PROJECT_SOURCE_DIR}/lib/ncnn  # ncnn lib path
)

### opencv
# set(OpenCV_DIR /usr/local/opencv-3.4.7/share/OpenCV)
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
else(OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV library not found")
endif(OpenCV_FOUND)

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR})
aux_source_directory(${CMAKE_SOURCE_DIR} SRC)

# include_directories(${CMAKE_SOURCE_DIR}/common)
aux_source_directory(${CMAKE_SOURCE_DIR}/common COMMON_SRC)

add_executable(${PROJECT_NAME} ${COMMON_SRC} ${SRC})

target_link_libraries(${PROJECT_NAME}
                      ascendcl acl_dvpp stdc++ 
                      dl rt)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} libncnn.a pthread -fopenmp)
target_link_libraries(${PROJECT_NAME} -lstdc++fs)
