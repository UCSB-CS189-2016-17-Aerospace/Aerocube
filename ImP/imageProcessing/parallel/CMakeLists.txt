# http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_gcc_cmake/linux_gcc_cmake.html
cmake_minimum_required(VERSION 3.0)
project(MarkerDetectParGPU)
find_package(OpenCV REQUIRED)
add_executable(MarkerDetectParGPU markerDetectParGPU.cpp)
target_link_libraries(MarkerDetectParGPU ${OpenCV_LIBS})