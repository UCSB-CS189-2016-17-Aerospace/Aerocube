#include "markerDetectParGPU.h"
#include "opencv2/cudawarping.hpp"

void warpPerspectiveWrapper(cv::InputArray _src, cv::OutputArray _dst, cv::InputArray _M, cv::Size dsize, int flags) {
    int i = 0;
    printf("ayy");
    cv::cuda::warpPerspective(_src, _dst, _M, dsize, flags);
}

int main() {
    return 1;
}