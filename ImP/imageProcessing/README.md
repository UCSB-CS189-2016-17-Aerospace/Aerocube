# Image Processing (ImP) Module
## Purpose
1. Given an image, identify AeroCubes and their IDs in the image
2. Provide information about the attitude and position of an identified AeroCube

## Public Interface
### ImageProcessor
### Exceptions
* **AeroCubeMarkerAttributeError**
* **IDOutOfDictionaryBoundError**

## Aruco Usage
### Identifying Markers
```
detectMarkers(...)
       detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedImgPoints]]]]) -> corners, ids, rejectedImgPoints
```
Parameters:
* **image** - image from which we want to detect markers
* **dictionary** - dictionary of marker IDs we will use to identify markers

Return values:
* **corners** - array of 2-D matrices identifying the x,y coordinates of the corners in the image
* **ids** - array of marker IDs based on the dictionary argument, respective to the corners
* **rejectedImgPoints** - rejected candidate markers

### Estimating Pose
```
estimatePoseSingleMarkers(...)
        estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs[, rvecs[, tvecs]]) -> rvecs, tvecs
```
Parameters:
* **corners** - array of 2-D matrices identifying the markers; returned by aruco.detectMarkers(...)
* **markerLength** - size of the marker side in meters or any other unit; the unit used will be used in rvec and tvec (see below)
* **cameraMatrix** - camera calibration 3x3 matrix
* **distCoeffs** - camera calibratian parameter

Return values:
* **rvecs** - rotation vectors for the given corners
* **tvecs** - translation vectors for the given corners
