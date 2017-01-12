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
    _detectCandidates(grey, candidates, contours, _params);
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
Verify [approxPolyDP](http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#approxpolydp) returns with 4 curves & is convex.
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
Parallel CPU implementation of the function above, given a range of scales, it will threshold and find contours.
```

   // this is the parallel call for the previous commented loop (result is equivalent)
   parallel_for_(Range(0, nScales), DetectInitialCandidatesParallel(&grey, &candidatesArrays,
                                                                    &contoursArrays, params));
```
Aggregating the contours from each scale into 1 vector of vectors.
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
```
### _reorderCandidatesCorners 
```
/**
 * @brief Assure order of candidate corners is clockwise direction
 */
static void _reorderCandidatesCorners(vector< vector< Point2f > > &candidates) {

   for(unsigned int i = 0; i < candidates.size(); i++) {
       double dx1 = candidates[i][1].x - candidates[i][0].x;
       double dy1 = candidates[i][1].y - candidates[i][0].y;
       double dx2 = candidates[i][2].x - candidates[i][0].x;
       double dy2 = candidates[i][2].y - candidates[i][0].y;
       double crossProduct = (dx1 * dy2) - (dy1 * dx2);

       if(crossProduct < 0.0) { // not clockwise direction
           swap(candidates[i][1], candidates[i][3]);
       }
   }
}
```
```
   /// 4. FILTER OUT NEAR CANDIDATE PAIRS
   _filterTooCloseCandidates(candidates, candidatesOut, contours, contoursOut,
                             _params->minMarkerDistanceRate);
}
```
### _filterTooCloseCandidates
```
/**
 * @brief Check candidates that are too close to each other and remove the smaller one
 */
static void _filterTooCloseCandidates(const vector< vector< Point2f > > &candidatesIn,
                                     vector< vector< Point2f > > &candidatesOut,
                                     const vector< vector< Point > > &contoursIn,
                                     vector< vector< Point > > &contoursOut,
                                     double minMarkerDistanceRate) {

   CV_Assert(minMarkerDistanceRate >= 0);
```
Comparing every possible pair, if considered too close, the pair of markers is put into nearCandidates .
```
   vector< pair< int, int > > nearCandidates;
   for(unsigned int i = 0; i < candidatesIn.size(); i++) {
       for(unsigned int j = i + 1; j < candidatesIn.size(); j++) {

           int minimumPerimeter = min((int)contoursIn[i].size(), (int)contoursIn[j].size() );

           // fc is the first corner considered on one of the markers, 4 combinations are possible
           for(int fc = 0; fc < 4; fc++) {
               double distSq = 0;
               for(int c = 0; c < 4; c++) {
                   // modC is the corner considering first corner is fc
                   int modC = (c + fc) % 4;
                   distSq += (candidatesIn[i][modC].x - candidatesIn[j][c].x) *
                                 (candidatesIn[i][modC].x - candidatesIn[j][c].x) +
                             (candidatesIn[i][modC].y - candidatesIn[j][c].y) *
                                 (candidatesIn[i][modC].y - candidatesIn[j][c].y);
               }
               distSq /= 4.;

               // if mean square distance is too low, remove the smaller one of the two markers
               double minMarkerDistancePixels = double(minimumPerimeter) * minMarkerDistanceRate;
               if(distSq < minMarkerDistancePixels * minMarkerDistancePixels) {
                   nearCandidates.push_back(pair< int, int >(i, j));
                   break;
               }
           }
       }
   }
```
Choses the smaller one in the pairs to be marked and later removed. 
```
   // mark smaller one in pairs to remove
   vector< bool > toRemove(candidatesIn.size(), false);
   for(unsigned int i = 0; i < nearCandidates.size(); i++) {
       // if one of the marker has been already markerd to removed, dont need to do anything
       if(toRemove[nearCandidates[i].first] || toRemove[nearCandidates[i].second]) continue;
       size_t perimeter1 = contoursIn[nearCandidates[i].first].size();
       size_t perimeter2 = contoursIn[nearCandidates[i].second].size();
       if(perimeter1 > perimeter2)
           toRemove[nearCandidates[i].second] = true;
       else
           toRemove[nearCandidates[i].first] = true;
   }
```
If not marked for removal add candidates to candidatesIn
```
   // remove extra candidates
   candidatesOut.clear();
   unsigned long totalRemaining = 0;
   for(unsigned int i = 0; i < toRemove.size(); i++)
       if(!toRemove[i]) totalRemaining++;
   candidatesOut.resize(totalRemaining);
   contoursOut.resize(totalRemaining);
   for(unsigned int i = 0, currIdx = 0; i < candidatesIn.size(); i++) {
       if(toRemove[i]) continue;
       candidatesOut[currIdx] = candidatesIn[i];
       contoursOut[currIdx] = contoursIn[i];
       currIdx++;
   }
}
```

```
    /// STEP 2: Check candidate codification (identify markers)
    _identifyCandidates(grey, candidates, contours, _dictionary, candidates, ids, _params,
                        _rejectedImgPoints);
```
### _identifyCandidates 
```
/**
* @brief Identify square candidates according to a marker dictionary
*/
static void _identifyCandidates(InputArray _image, vector< vector< Point2f > >& _candidates,
                               InputArrayOfArrays _contours, const Ptr<Dictionary> &_dictionary,
                               vector< vector< Point2f > >& _accepted, vector< int >& ids,
                               const Ptr<DetectorParameters> &params,
                               OutputArrayOfArrays _rejected = noArray()) {

   int ncandidates = (int)_candidates.size();

   vector< vector< Point2f > > accepted;
   vector< vector< Point2f > > rejected;

   CV_Assert(_image.getMat().total() != 0);

   Mat grey;
   _convertToGrey(_image.getMat(), grey);

   vector< int > idsTmp(ncandidates, -1);
   vector< char > validCandidates(ncandidates, 0);

   //// Analyze each of the candidates
   // for (int i = 0; i < ncandidates; i++) {
   //    int currId = i;
   //    Mat currentCandidate = _candidates.getMat(i);
   //    if (_identifyOneCandidate(dictionary, grey, currentCandidate, currId, params)) {
   //        validCandidates[i] = 1;
   //        idsTmp[i] = currId;
   //    }
   //}

   // this is the parallel call for the previous commented loop (result is equivalent)
   parallel_for_(Range(0, ncandidates),
                 IdentifyCandidatesParallel(grey, _candidates, _contours, _dictionary, idsTmp,
                                            validCandidates, params));

   for(int i = 0; i < ncandidates; i++) {
       if(validCandidates[i] == 1) {
           accepted.push_back(_candidates[i]);
           ids.push_back(idsTmp[i]);
       } else {
           rejected.push_back(_candidates[i]);
       }
   }

   // parse output
   _accepted = accepted;

   if(_rejected.needed()) {
       _copyVector2Output(rejected, _rejected);
   }
}
```
```

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
