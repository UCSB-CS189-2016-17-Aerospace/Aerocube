#!/usr/bin/env bash
# Point to opencv_contrib modules directory
# Build with CUDA
# Disable precompiled headers to save space
# Disable tests, performance tests, and examples
# Disable certain opencv modules to save space
# Disable most opencv_contrib modules (all but Aruco)
cmake -DOPENCV_EXTRA_MODULES_PATH=/usr/local/lib/opencv_contrib/modules/ \
    -D WITH_CUDA=ON -D CUDA_FAST_MATH=1 \
    -D ENABLE_PRECOMPILED_HEADERS=OFF \
    -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF \
    -D BUILD_opencv_java=OFF \
    -D BUILD_opencv_videostab=OFF \
    -DBUILD_opencv_bgsegm=OFF -DBUILD_opencv_bioinspired=OFF -DBUILD_opencv_ccalib=OFF -DBUILD_opencv_cnn_3dobj=OFF -DBUILD_opencv_contrib_world=OFF -DBUILD_opencv_cvv=OFF -DBUILD_opencv_datasets=OFF -DBUILD_opencv_dnn=OFF -DBUILD_opencv_dnns_easily_fooled=OFF -DBUILD_opencv_dpm=OFF -DBUILD_opencv_face=OFF -DBUILD_opencv_fuzzy=OFF -DBUILD_opencv_hdf=OFF -DBUILD_opencv_line_descriptor=OFF -DBUILD_opencv_matlab=OFF -DBUILD_opencv_optflow=OFF -DBUILD_opencv_plot=OFF -DBUILD_opencv_README.md=OFF -DBUILD_opencv_reg=OFF -DBUILD_opencv_rgbd=OFF -DBUILD_opencv_saliency=OFF -DBUILD_opencv_sfm=OFF -DBUILD_opencv_stereo=OFF -DBUILD_opencv_structured_light=OFF -DBUILD_opencv_surface_matching=OFF -DBUILD_opencv_text=OFF -DBUILD_opencv_tracking=OFF -DBUILD_opencv_viz=OFF -DBUILD_opencv_xfeatures2d=OFF -DBUILD_opencv_ximgproc=OFF -DBUILD_opencv_xobjdetect=OFF -DBUILD_opencv_xphoto=OFF ..
