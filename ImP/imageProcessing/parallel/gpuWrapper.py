import os
import ctypes
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray
from pycuda.compiler import SourceModule

# Ctypes Wrappers for CUDA Libraries

_SO_DIR = os.path.dirname(__file__)
_MARKER_DETECT_PAR_GPU = ctypes.cdll.LoadLibrary(os.path.join(_SO_DIR, 'libMarkerDetectParGPU.so'))

# Initialize warpPerspective


class CV_SIZE(ctypes.Structure):
    _fields_ = [
        ('height', ctypes.c_float),
        ('width', ctypes.c_float)
    ]


def _initialize_warp_perspective():
    """
    cv::cuda::warpPerspective function signature
    * http://docs.opencv.org/trunk/db/d29/group__cudawarping.html#ga7a6cf95065536712de6b155f3440ccff
    :return:
    """
    _func = _MARKER_DETECT_PAR_GPU.warpPerspectiveWrapper
    _func.restype = ctypes.c_int32
    _func.argtypes = [ctypes.c_void_p,
                      ctypes.c_void_p,
                      ctypes.c_void_p,
                      ctypes.POINTER(CV_SIZE),
                      ctypes.c_int32]
    return _func


# Assign wrapped functions to private module variables
_func_warp_perspective = _initialize_warp_perspective()


def _cuda_warp_perspective(src, M, dsize, flags=cv2.INTER_NEAREST):
    # TODO: convert src and dst to GpuMat
    # Get the warpPerspective function
    warpPerspective = _func_warp_perspective
    # Convert src, M to ctype-friendly for mat
    src_gpu = pycuda.gpuarray.to_gpu(src.astype(np.float32))
    M_gpu = pycuda.gpuarray.to_gpu(M.astype(np.float32))
    # Create dst as ctype-friendly format
    dst_gpu = pycuda.gpuarray.to_gpu(np.zeros(src.shape, dtype=np.float32))
    # Convert dsize to specified format
    cv_size = CV_SIZE(*dsize)
    warpPerspective(int(src_gpu.gpudata),
                    int(dst_gpu.gpudata),
                    int(M_gpu.gpudata),
                    cv_size,
                    flags)
    # return dst_gpu.astype(np.float32)