#ifndef AEROCUBE_MARKERDETECTPARGPU_H
#define AEROCUBE_MARKERDETECTPARGPU_H

#include <opencv2/cudawarping.hpp>

// Use extern "C" to prevent C++ name-mangling, and allow easy find through Python ctypes
extern "C" {
    void warpPerspectiveWrapper(std::vector< std::vector <int> > *_src,
                                std::vector< std::vector <int> > *_dst,
                                std::vector< std::vector <int> > *_M,
                                cv::Size dsize,
                                int flags);
}


#endif //AEROCUBE_MARKERDETECTPARGPU_H
