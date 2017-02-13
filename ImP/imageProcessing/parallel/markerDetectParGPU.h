#ifndef AEROCUBE_MARKERDETECTPARGPU_H
#define AEROCUBE_MARKERDETECTPARGPU_H

#include <opencv2/cudawarping.hpp>

// Use extern "C" to prevent C++ name-mangling, and allow easy find through Python ctypes
extern "C" {
    void warpPerspectiveWrapper(cv::InputArray _src, cv::OutputArray _dst, cv::InputArray _M, cv::Size dsize, int flags);
}


#endif //AEROCUBE_MARKERDETECTPARGPU_H
