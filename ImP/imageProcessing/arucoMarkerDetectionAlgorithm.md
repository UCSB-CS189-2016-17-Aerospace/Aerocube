### Marker Detection Algorithm
Marker detection is publicly available through the following function in ```aruco.cpp```.
```
void detectMarkers(InputArray _image, const Ptr<Dictionary> &_dictionary, OutputArrayOfArrays _corners,
                   OutputArray _ids, const Ptr<DetectorParameters> &_params,
                   OutputArrayOfArrays _rejectedImgPoints) {
```
Assert that the image is not empty 
```

    CV_Assert(!_image.empty());
```
For interpretation of data types, such as CV_8UC1, refer to this [link](http://docs.opencv.org/2.4/modules/core/doc/basic_structures.html#datatype)

```
    Mat grey;
    _convertToGrey(_image.getMat(), grey);
```

Private method is called that checks to see if the matrix is a 3-channels image, and converts it to a 1-channel image (grayscale).

```
/** * @brief Convert input image to gray if it is a 3-channels image
 */
static void _convertToGrey(InputArray _in, OutputArray _out) {

   CV_Assert(_in.getMat().channels() == 1 || _in.getMat().channels() == 3);

   _out.create(_in.getMat().size(), CV_8UC1);
   if(_in.getMat().type() == CV_8UC3)
       cvtColor(_in.getMat(), _out.getMat(), COLOR_BGR2GRAY);
   else
       _in.getMat().copyTo(_out);
}
```

Point class found [here](http://docs.opencv.org/2.4/modules/core/doc/basic_structures.html#point)

```
    /// STEP 1: Detect marker candidates
    vector< vector< Point2f > > candidates;
    vector< vector< Point > > contours;
    vector< int > ids;
```

```
/**
* @brief Detect square candidates in the input image
*/
static void _detectCandidates(InputArray _image, vector< vector< Point2f > >& candidatesOut,
                             vector< vector< Point > >& contoursOut, const Ptr<DetectorParameters> &_params) {

   Mat image = _image.getMat();
   CV_Assert(image.total() != 0);

   /// 1. CONVERT TO GRAY
   Mat grey;
   _convertToGrey(image, grey);

   vector< vector< Point2f > > candidates;
   vector< vector< Point > > contours;
   /// 2. DETECT FIRST SET OF CANDIDATES
   _detectInitialCandidates(grey, candidates, contours, _params);

   /// 3. SORT CORNERS
   _reorderCandidatesCorners(candidates);

   /// 4. FILTER OUT NEAR CANDIDATE PAIRS
   _filterTooCloseCandidates(candidates, candidatesOut, contours, contoursOut,
                             _params->minMarkerDistanceRate);
}
```

```
    _detectCandidates(grey, candidates, contours, _params);
```
```
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
