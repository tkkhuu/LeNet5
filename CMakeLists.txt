cmake_minimum_required(VERSION 3.0)

project(lenet5_rt)

find_package(CUDA 10.2 REQUIRED)

# CMake path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# first try to find cuda and tensorrt-related stuff (nvinfer, onnx, etc)
find_package(TensorRT REQUIRED 7.0)
if (TENSORRT_FOUND)
  message(STATUS "TensorRT IncludeDir ${TensorRT_INCLUDE_DIRS}")
  message(STATUS "TensorRT Version ${TensorRT_VERSION_STRING}")
  set(TENSORRT_FOUND ON)
else()
  message("TensorRT NOT FOUND\n")
  set(TENSORRT_FOUND OFF)
endif (TENSORRT_FOUND)

include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
)

add_library(lenet5_rt src/lenet5_rt.cpp)
add_executable(run_lenet5_inf src/main.cpp)

message(STATUS "TensorRT Library is ${TensorRT_LIBRARY}")

target_link_libraries(lenet5_rt
    ${TensorRT_LIBRARY}
    ${CUDA_LIBRARIES}
    "/usr/lib/x86_64-linux-gnu/libnvparsers.so"
)
target_link_libraries(run_lenet5_inf
    lenet5_rt
)