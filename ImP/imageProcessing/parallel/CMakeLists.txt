# http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_gcc_cmake/linux_gcc_cmake.html
# Specify minimum required version of CMake
cmake_minimum_required(VERSION 3.0)
# Specify project name
project(MarkerDetectParGPU)
# Find OpenCV
find_package(OpenCV REQUIRED)
# No need to add executable
# add_executable(MarkerDetectParGPU markerDetectParGPU.cpp)
# Add shared object (.so) library
add_library(MarkerDetectParGPU SHARED markerDetectParGPU.cpp)
# Link necessary libraries
target_link_libraries(MarkerDetectParGPU ${OpenCV_LIBS})