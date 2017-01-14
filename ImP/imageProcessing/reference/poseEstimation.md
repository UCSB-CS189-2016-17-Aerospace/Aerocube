# Pose Estimation
## Aruco Pose Estimation Algorithm
Aruco allows for pose estimation of a marker through the following public function (in C++).
```
void estimatePoseSingleMarkers(InputArrayOfArrays _corners, float markerLength,
                               InputArray _cameraMatrix, InputArray _distCoeffs,
                               OutputArray _rvecs, OutputArray _tvecs)
```
The function's serial implementation loops through the markers and calls OpenCV's [solvePnP](http://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#solvepnp) function.
```
for (int i = 0; i < nMarkers; i++) {
    solvePnP(markerObjPoints, _corners.getMat(i), _cameraMatrix, _distCoeffs,
             _rvecs.getMat(i), _tvecs.getMat(i))
```
The function returns two arrays (of length nMarkers) of rotation and translation vectors for each marker.
```
_rvecs.create(nMarkers, 1, CV_64FC3);
_tvecs.create(nMarkers, 1, CV_64FC3);
```
These are of type CV_64FC3, meaning they are 64 bits deep, float coordinates with 3 values.

## Quaternion
