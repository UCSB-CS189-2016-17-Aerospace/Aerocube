# Aerocube-ImP
This library will handle image processing, image detection algorithms, and other computer vision voodoo-features for detecting our CubeSatellite.

# Dependencies
## Chilitag Dependencies
See source Chilitags repository for reference: <https://github.com/chili-epfl/chilitags>

Install OpenCV Libraries and CMake

`sudo apt-get install libopencv-dev cmake`

# Instructions
To build the project:
```
mkdir build && cd build
cmake ..
make
```

To build and execute chilitags samples:

```
mkdir build && cd build
cmake -DWITH_SAMPLES=ON ..
make
./lib/chilitags/samples/detect-live

```



# Feature List
  * ImP - 0.0.0
    * CMake file for OpenCV library compilation
    * Base Hello World Application in OpenCV

  * ImP - 0.0.1 - (10/23-10/29)
    * Chilitags is imported
    * Chilitags dependency is compiling with a global CMake
    * Fiducial Marker detection samples running

# Troubleshooting
If you encounter any errors written as "error: No member named ... in namespace cv", then you may be running OpenCV3, rather than OpenCV2. 
To resolve any potential 'No member named ... in namespace cv errors', please include the correct header files, as some functions may be moved during the OpenCV version bump to 3.0.
For example, drawings functions were moved to <opencv2/imgproc.hpp> in opencv3.


# Licenses
Credit goes to Chilitags for providing Fiducial Marker tracking software.

Chilitags: Robust Fiducial Markers for Augmented Reality. Q. Bonnard, S. Lemaignan, G. Zufferey, A. Mazzei, S. Cuendet, N. Li, P. Dillenbourg. CHILI, EPFL, Switzerland. http://chili.epfl.ch/software. 2013.


# Tests

## Run Tests
```
# Assumes you are in the project root folder
mkdir build && cd build
cmake -DTESTS=ON ..
make runTests # runs the runTests test-suite
./runTests
```

## Test Dependencies
This repository uses GoogleTest (GTest) on Ubuntu for testing.

To run tests, packages and linked libraries are required on your host machine (not this repository).

```
# Install the GTest development package
sudo apt-get install libgtest-dev

# Find the source files at /usr/src/gtest/
# This repo assumes that cmake is installed. So `sudo apt-get install cmake` is not required if already present.

cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make

# copy or symlink libgtest.a and libgtest_main.a to your /usr/lib folder
sudo cp *.a /usr/lib

# FIN - GTest should be installed on the machine, so now CMake should be able to  find the GTest

```

