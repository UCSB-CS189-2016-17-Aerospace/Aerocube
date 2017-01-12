# Marker Detection Algorithm
Marker detection is publicly available through the following function in ```aruco.cpp```.

```
void detectMarkers(InputArray _image, const Ptr<Dictionary> &_dictionary, OutputArrayOfArrays _corners,
                   OutputArray _ids, const Ptr<DetectorParameters> &_params,
                   OutputArrayOfArrays _rejectedImgPoints) {
```

## Step 0: Asserts image is not empty, then converts it to grayscale (if not already).

Assert that the image is not empty 
```

    CV_Assert(!_image.empty());
```
For interpretation of data types, such as CV_8UC1, refer to this [link](http://docs.opencv.org/2.4/modules/core/doc/basic_structures.html#datatype)

```
    Mat grey;
    _convertToGrey(_image.getMat(), grey);
```

## Step 1: Creates the 

Creates 2-dimensional arrray of float coordinates to keep track of candidate markers. Candidates are potential markers that still have to be validated by the algorithm.
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
```
### _detectInitialCandidates
```
/**
* @brief Initial steps on finding square candidates
*/
static void _detectInitialCandidates(const Mat &grey, vector< vector< Point2f > > &candidates,
                                    vector< vector< Point > > &contours,
                                    const Ptr<DetectorParameters> &params) {

   CV_Assert(params->adaptiveThreshWinSizeMin >= 3 && params->adaptiveThreshWinSizeMax >= 3);
   CV_Assert(params->adaptiveThreshWinSizeMax >= params->adaptiveThreshWinSizeMin);
   CV_Assert(params->adaptiveThreshWinSizeStep > 0);

   
```
More information about scale space, click [here](https://en.wikipedia.org/wiki/Scale_space)

```
   // number of window sizes (scales) to apply adaptive thresholding
   int nScales =  (params->adaptiveThreshWinSizeMax - params->adaptiveThreshWinSizeMin) /
                     params->adaptiveThreshWinSizeStep + 1;

   vector< vector< vector< Point2f > > > candidatesArrays((size_t) nScales);
   vector< vector< vector< Point > > > contoursArrays((size_t) nScales);
```
[Threshold](https://en.wikipedia.org/wiki/Thresholding_(image_processing)) the image at different scales to find different markers.

```
   ////for each value in the interval of thresholding window sizes
   // for(int i = 0; i < nScales; i++) {
   //    int currScale = params.adaptiveThreshWinSizeMin + i*params.adaptiveThreshWinSizeStep;
   //    // treshold
   //    Mat thresh;
   //    _threshold(grey, thresh, currScale, params.adaptiveThreshConstant);
   //    // detect rectangles
   //    _findMarkerContours(thresh, candidatesArrays[i], contoursArrays[i],
   // params.minMarkerPerimeterRate,
   //                        params.maxMarkerPerimeterRate, params.polygonalApproxAccuracyRate,
   //                        params.minCornerDistance, params.minDistanceToBorder);
   //}
```
### _findMarkerContours

```
/**
 * @brief Given a tresholded image, find the contours, calculate their polygonal approximation
 * and take those that accomplish some conditions
 */
static void _findMarkerContours(InputArray _in, vector< vector< Point2f > > &candidates,
                               vector< vector< Point > > &contoursOut, double minPerimeterRate,
                               double maxPerimeterRate, double accuracyRate,
                               double minCornerDistanceRate, int minDistanceToBorder) {

   CV_Assert(minPerimeterRate > 0 && maxPerimeterRate > 0 && accuracyRate > 0 &&
             minCornerDistanceRate >= 0 && minDistanceToBorder >= 0);

   // calculate maximum and minimum sizes in pixels
   unsigned int minPerimeterPixels =
       (unsigned int)(minPerimeterRate * max(_in.getMat().cols, _in.getMat().rows));
   unsigned int maxPerimeterPixels =
       (unsigned int)(maxPerimeterRate * max(_in.getMat().cols, _in.getMat().rows));

   Mat contoursImg;
   _in.getMat().copyTo(contoursImg);
   vector< vector< Point > > contours;
   findContours(contoursImg, contours, RETR_LIST, CHAIN_APPROX_NONE);
```
```
   // now filter list of contours
   for(unsigned int i = 0; i < contours.size(); i++) {
       // check perimeter
       if(contours[i].size() < minPerimeterPixels || contours[i].size() > maxPerimeterPixels)
           continue;
```
```
       // check is square and is convex
       vector< Point > approxCurve;
       approxPolyDP(contours[i], approxCurve, double(contours[i].size()) * accuracyRate, true);
       if(approxCurve.size() != 4 || !isContourConvex(approxCurve)) continue;
```
```
       // check min distance between corners
       double minDistSq =
           max(contoursImg.cols, contoursImg.rows) * max(contoursImg.cols, contoursImg.rows);
       for(int j = 0; j < 4; j++) {
           double d = (double)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) *
                          (double)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) +
                      (double)(approxCurve[j].y - approxCurve[(j + 1) % 4].y) *
                          (double)(approxCurve[j].y - approxCurve[(j + 1) % 4].y);
           minDistSq = min(minDistSq, d);
       }
       double minCornerDistancePixels = double(contours[i].size()) * minCornerDistanceRate;
       if(minDistSq < minCornerDistancePixels * minCornerDistancePixels) continue;
```
```
       // check if it is too near to the image border
       bool tooNearBorder = false;
       for(int j = 0; j < 4; j++) {
           if(approxCurve[j].x < minDistanceToBorder || approxCurve[j].y < minDistanceToBorder ||
              approxCurve[j].x > contoursImg.cols - 1 - minDistanceToBorder ||
              approxCurve[j].y > contoursImg.rows - 1 - minDistanceToBorder)
               tooNearBorder = true;
       }
       if(tooNearBorder) continue;

       // if it passes all the test, add to candidates vector
       vector< Point2f > currentCandidate;
       currentCandidate.resize(4);
       for(int j = 0; j < 4; j++) {
           currentCandidate[j] = Point2f((float)approxCurve[j].x, (float)approxCurve[j].y);
       }
       candidates.push_back(currentCandidate);
       contoursOut.push_back(contours[i]);
   }
```
```
}
```

```

   // this is the parallel call for the previous commented loop (result is equivalent)
   parallel_for_(Range(0, nScales), DetectInitialCandidatesParallel(&grey, &candidatesArrays,
                                                                    &contoursArrays, params));
```
```
   // join candidates
   for(int i = 0; i < nScales; i++) {
       for(unsigned int j = 0; j < candidatesArrays[i].size(); j++) {
           candidates.push_back(candidatesArrays[i][j]);
           contours.push_back(contoursArrays[i][j]);
       }
   }
}
```

```
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

## Appendix 

### Commonly refered to functions

### _convertToGrey

Private method that checks to see if the matrix is a 3-channels image, and converts it to a 1-channel image (grayscale).

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

### _threshold

Explanation of what adaptiveThreshold does can be found [here](http://docs.opencv.org/3.0-beta/modules/imgproc/doc/miscellaneous_transformations.html)
```
/**
 * @brief Threshold input image using adaptive thresholding
 */
static void _threshold(InputArray _in, OutputArray _out, int winSize, double constant) {

   CV_Assert(winSize >= 3);
   if(winSize % 2 == 0) winSize++; // win size must be odd
   adaptiveThreshold(_in, _out, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, winSize, constant);
}
```


### DetectorParameters::DetectorParameters()
```
/**
 *
 */
DetectorParameters::DetectorParameters()
   : adaptiveThreshWinSizeMin(3),  
     adaptiveThreshWinSizeMax(23),
     adaptiveThreshWinSizeStep(10),
     adaptiveThreshConstant(7),
     minMarkerPerimeterRate(0.03),
     maxMarkerPerimeterRate(4.),
     polygonalApproxAccuracyRate(0.03),
     minCornerDistanceRate(0.05),
     minDistanceToBorder(3),
     minMarkerDistanceRate(0.05),
     doCornerRefinement(false),
     cornerRefinementWinSize(5),
     cornerRefinementMaxIterations(30),
     cornerRefinementMinAccuracy(0.1),
     markerBorderBits(1),
     perspectiveRemovePixelPerCell(4),
     perspectiveRemoveIgnoredMarginPerCell(0.13),
     maxErroneousBitsInBorderRate(0.35),
     minOtsuStdDev(5.0),
     errorCorrectionRate(0.6) {}
```

### adaptive threshold 
* **adaptiveThreshWinSizeMin** Smallest size for the window used in thresholding must be at least 3. Affects the first scale and possibly how many scales will be created when looking at different thresholded images. 
* **adaptiveThreshWinSizeMax** Max size for the window used in thresholding must be at least 3, but bigger than the min. Also affects how many scales will be created when looking at different thresholded images.
* **adaptiveThreshWinSizeStep** Size of the step for the window to move through. Also affects how many scales will be created when looking at different thresholded images.
* **adaptiveThreshConstant** Constant that is used to subtract off the threshold mean.
