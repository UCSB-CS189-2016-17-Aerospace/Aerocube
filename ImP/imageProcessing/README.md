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

### Marker Detection Algorithm
Marker detection is publicy available through the following function in ```aruco.cpp```.
```
void detectMarkers(InputArray _image, const Ptr<Dictionary> &_dictionary, OutputArrayOfArrays _corners,
                   OutputArray _ids, const Ptr<DetectorParameters> &_params,
                   OutputArrayOfArrays _rejectedImgPoints) {
     CV_Assert(!_image.empty());

    Mat grey;
    _convertToGrey(_image.getMat(), grey);

    /// STEP 1: Detect marker candidates
    vector< vector< Point2f > > candidates;
    vector< vector< Point > > contours;
    vector< int > ids;
    _detectCandidates(grey, candidates, contours, _params);

    /// STEP 2: Check candidate codification (identify markers)
    _identifyCandidates(grey, candidates, contours, _dictionary, candidates, ids, _params,
                        _rejectedImgPoints);

    /// STEP 3: Filter detected markers;
    _filterDetectedMarkers(candidates, ids);

    // copy to output arrays
    _copyVector2Output(candidates, _corners);
    Mat(ids).copyTo(_ids);

    /// STEP 4: Corner refinement
    if(_params->doCornerRefinement) {
        CV_Assert(_params->cornerRefinementWinSize > 0 && _params->cornerRefinementMaxIterations > 0 &&
                  _params->cornerRefinementMinAccuracy > 0);

        //// do corner refinement for each of the detected markers
        // for (unsigned int i = 0; i < _corners.cols(); i++) {
        //    cornerSubPix(grey, _corners.getMat(i),
        //                 Size(params.cornerRefinementWinSize, params.cornerRefinementWinSize),
        //                 Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
        //                                            params.cornerRefinementMaxIterations,
        //                                            params.cornerRefinementMinAccuracy));
        //}

        // this is the parallel call for the previous commented loop (result is equivalent)
        parallel_for_(Range(0, _corners.cols()),
                      MarkerSubpixelParallel(&grey, _corners, _params));
    }
```
