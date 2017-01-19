# Pose Estimation
## Pose Representation
There are a number of different ways to represent pose or, more broadly, rotation. In particular, [rotation in three dimensions](https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions) can take the form of:
* Rotation matrix - an orthogonal, 3 x 3 matrix composed of three unit vectors; has 3 degrees of freedom
* [Euler axis and angle (rotation vector)](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation) - three-dimensional unit vector with angle θ; 3 degrees of freedom, and more compact than rotation matrix
* Rodrigues parameters - expressed in terms of the Euler axis and angle, (i.e., rotation vector)
* [Quaternions](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation) - form a four-dimensional vector space; often favored to represent rotations due to numerous advantages; normalizing is less computationally expensive than normalizing rotation matrices

## Aruco Pose Estimation Algorithm
#### Python Function Signature
Aruco allows for pose estimation of a marker through the following function.
```
estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs[, rvecs[, tvecs]]) -> rvecs, tvecs
```
The arguments are:
* **corners** - array of corner points of markers (four corners per marker)
* **markerLength** - physical length of marker; this determines the units/scaling of the returned rvecs and tvecs
* **cameraMatrix** - the [intrinsic camera matrix](https://en.wikipedia.org/wiki/Camera_resectioning#Intrinsic_parameters) of the camera used to take the image; can be referenced from the CameraCalibration module
* **distCoeffs** - [distortion coefficients](http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html) of the camera used to take the image; can be referenced from the CameraCalibration module

The return values are:
* **rvecs** - array of rotation vectors of each marker; a rotation matrix that has been converted into rotation vector according to the [Rodrigues parameters](https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rodrigues_parameters_and_Gibbs_representation)
* **tvecs** - array of translation vectors of each marker; recall that the numbers are scaled off of the original **markerLength** argument

#### Function Implementation
The original function in C++ is defined as such.
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
#### rvec
solvePnP's outputs of ```rvec``` is a rotation matrix that has been converted to a rotation vector by using [Rodrigues()](http://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#rodrigues).

#### tvec
* [OpenCV: solvePnP tvec units and axes directions](http://stackoverflow.com/questions/17423302/opencv-solvepnp-tvec-units-and-axes-directions)

## Quaternion
#### Information on Quaternions
* [Euler to Quaternion - Sample Orientations](http://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToQuaternion/steps/index.htm)

#### Translating to the Quaternion
* [Wikipedia: Euler axis/angle to quaternion](https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Euler_axis.2Fangle_.E2.86.94_quaternion)
* [Euclidean Space: AxisAngle to Quaternion](http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/)
