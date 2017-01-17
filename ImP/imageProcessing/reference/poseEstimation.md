# Pose Estimation
## Pose Representation
There are a number of different ways to represent pose or, more broadly, rotation. In particular, [rotation in three dimensions](https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions) can take the form of:
* Rotation matrix - an orthogonal, 3 x 3 matrix composed of three unit vectors; has 3 degrees of freedom
* Euler axis and angle (rotation vector) - three-dimensional unit vector with angle θ; 3 degrees of freedom, and more compact than rotation matrix
* Rodrigues parameters - 
* [Quaternions](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation) - form a four-dimensional vector space; often favored to represent rotations due to numerous advantages; normalizing is less computationally expensive than normalizing rotation matrices

## Aruco Pose Estimation Algorithm
Aruco allows for pose estimation of a marker through the following public function (in C++).
```
void estimatePoseSingleMarkers(InputArrayOfArrays _corners, float markerLength,
                               InputArray _cameraMatrix, InputArray _distCoeffs,
                               OutputArray _rvecs, OutputArray _tvecs)
```
The function returns two arrays (of length nMarkers) of rotation and translation vectors for each marker.
```
_rvecs.create(nMarkers, 1, CV_64FC3);
_tvecs.create(nMarkers, 1, CV_64FC3);
```
These are arrays with as many elements as there are markers. Each element is of type [CV_64FC3](http://docs.opencv.org/trunk/d0/d3a/classcv_1_1DataType.html), meaning they are 3-element floating-point tuples of bit-depth 64.

The function's serial implementation loops through the markers and calls OpenCV's [solvePnP](http://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#solvepnp) function.
```
for (int i = 0; i < nMarkers; i++) {
    solvePnP(markerObjPoints, _corners.getMat(i), _cameraMatrix, _distCoeffs,
             _rvecs.getMat(i), _tvecs.getMat(i))
```
### rvec
solvePnP's outputs of ```rvec``` is a rotation matrix that has been converted to a rotation vector by using [Rodrigues()](http://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#rodrigues).

Because the rvec output of solvePnP's seems to be 
 
### tvec
* [OpenCV: solvePnP tvec units and axes directions](http://stackoverflow.com/questions/17423302/opencv-solvepnp-tvec-units-and-axes-directions)

## Quaternion
