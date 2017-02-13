#include <iostream>
#include "markerDetectParGPU.h"
#include "opencv2/cudawarping.hpp"

void warpPerspectiveWrapper(cv::InputArray _src, cv::OutputArray _dst, cv::InputArray _M, cv::Size dsize, int flags) {
    int i = 0;
    std::cout << "ayyyyyy" << std::endl;
    std::cout << "_src.isMat()" << std::endl;
    std::cout << _src.isMat() << std::endl;
    std::cout << "_src.isGpuMatVector()" << std::endl;
    std::cout << _src.isGpuMatVector() << std::endl;
    std::cout << "_src.getFlags()" << std::endl;
    std::cout << _src.getFlags() << std::endl;
    std::cout << "_src.rows()" << std::endl;
    std::cout << _src.rows() << std::endl;

    cv::cuda::warpPerspective(_src, _dst, _M, dsize, flags);
}

int main() {
    return 1;
}