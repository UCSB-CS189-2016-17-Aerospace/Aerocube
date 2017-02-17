#include <iostream>
#include <vector>
#include "markerDetectParGPU.h"
//#include "opencv2/mat.hpp"
#include "opencv2/cudawarping.hpp"

void warpPerspectiveWrapper(int *_src,
                            int *_dst,
                            int *_M,
                            cv::Size dsize,
                            int flags) {
    int i = 0;
    std::cout << "ayyyyyy" << std::endl;
    // Construct vectors from inputs
    // http://stackoverflow.com/questions/2434196/how-to-initialize-stdvector-from-c-style-array
    std::vector<int> src;
//    src.assign(_src, _src + dsize.)
    std::cout << _src->size() << std::endl;
    std::cout << (*_src)[0].size() << std::endl;

    cv::Mat src = cv::Mat(dsize.height, dsize.width, cv::CV_8UC1);
//
//    std::cout << "src.isMat()" << std::endl;
//    std::cout << src.isMat() << std::endl;

//    std::cout << "_src.isMat()" << std::endl;
//    std::cout << _src.isMat() << std::endl;
//    std::cout << "_src.isGpuMatVector()" << std::endl;
//    std::cout << _src.isGpuMatVector() << std::endl;
//    std::cout << "_src.getFlags()" << std::endl;
//    std::cout << _src.getFlags() << std::endl;
//    std::cout << "_src.rows()" << std::endl;
//    std::cout << _src.rows() << std::endl;

//    cv::cuda::warpPerspective(_src, _dst, _M, dsize, flags);
}

int main() {
    return 1;
}