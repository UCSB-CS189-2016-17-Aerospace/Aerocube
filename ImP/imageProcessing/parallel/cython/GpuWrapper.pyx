import numpy as np  # Import Python functions, attributes, submodules of numpy
cimport numpy as np  # Import C functions, attributes, submodules of numpy
from libcpp.string cimport string  # Import <string>
from libcpp cimport bool  # Get support for C++ bool
from libc.string cimport memcpy  # Import <string.h>, or <cstring>
from cpython.ref cimport PyObject

"""
References:
* http://makerwannabe.blogspot.com/2013/09/calling-opencv-functions-via-cython.html
* On converting PyObjects to OpenCV Mat objects
    * http://stackoverflow.com/questions/22736593/what-is-the-easiest-way-to-convert-ndarray-into-cvmat
    * http://stackoverflow.com/questions/12957492/writing-python-bindings-for-c-code-that-use-opencv/12972689#12972689
    * http://stackoverflow.com/questions/13745265/exposing-opencv-based-c-function-with-mat-numpy-conversion-to-python
    * https://github.com/Algomorph/pyboostcvconverter/blob/master/src/pyboost_cv3_converter.cpp
"""

# cdef extern from '/usr/local/lib/opencv/modules/python/src2/pycompat.hpp' namespace 'cv':
#     cdef PyObject* pyopencv_from(const Mat& m) except +
#     cdef bool pyopencv_to(PyObject* o, Mat& m) except +

cdef extern from 'pyopencv_converter.cpp':
    cdef PyObject* pyopencv_from(const Mat& m)
    cdef bool pyopencv_to(PyObject* o, Mat& m)

cdef extern from 'opencv2/imgproc.hpp' namespace 'cv':
    cdef enum InterpolationFlags:
        INTER_NEAREST = 0
    cdef enum ColorConversionCodes:
        COLOR_BGR2GRAY

cdef extern from 'opencv2/core/core.hpp':
    cdef int CV_8UC1
    cdef int CV_32FC1

cdef extern from 'opencv2/core/core.hpp' namespace 'cv':
    cdef cppclass Size_[T]:
        Size_() except +
        Size_(T width, T height) except +
        T width
        T height
    ctypedef Size_[int] Size2i
    ctypedef Size2i Size
    cdef cppclass Scalar[T]:
        Scalar() except +
        Scalar(T v0) except +

# Define C++ class Mat, InputArray, and OutputArray from core.hpp
cdef extern from 'opencv2/core/core.hpp' namespace 'cv':
    cdef cppclass Mat:
        Mat() except +
        void create(int, int, int) except +
        void* data
        int rows
        int cols

# Define C++ classes for CUDA
cdef extern from 'opencv2/core/cuda.hpp' namespace 'cv::cuda':
    cdef cppclass GpuMat:
        GpuMat() except +
        void upload(Mat arr) except +
        void download(Mat dst) const
    cdef cppclass Stream:
        Stream() except +

cdef extern from 'opencv2/cudawarping.hpp' namespace 'cv::cuda':
    cdef void warpPerspective(GpuMat src, GpuMat dst, Mat M, Size dsize, int flags, int borderMode, Scalar borderValue, Stream& stream)
    # Function using default values
    cdef void warpPerspective(GpuMat src, GpuMat dst, Mat M, Size dsize, int flags)

cdef extern from 'opencv2/imgproc.hpp' namespace 'cv':
    cdef void warpPerspective(Mat src, Mat dst, Mat M, Size dsize, int flags)

cdef extern from 'opencv2/cudaimgproc.hpp' namespace 'cv::cuda':
    cdef void cvtColor(GpuMat src, GpuMat dst, int code) except +

def cudaWarpPerspectiveWrapper(np.ndarray[np.uint8_t, ndim=2] _src,
                               np.ndarray[np.float32_t, ndim=2] _M,
                               _size_tuple,
                               int _flags=INTER_NEAREST):
    # Create GPU/device InputArray for src
    cdef Mat src_mat
    cdef GpuMat src_gpu
    pyopencv_to(<PyObject*> _src, src_mat)
    src_gpu.upload(src_mat)

    # Create CPU/host InputArray for M
    cdef Mat M_mat = Mat()
    pyopencv_to(<PyObject*> _M, M_mat)

    # Create Size object from size tuple
    # Note that size/shape in Python is handled in row-major-order -- therefore, width is [1] and height is [0]
    cdef Size size = Size(<int> _size_tuple[1], <int> _size_tuple[0])

    # Create empty GPU/device OutputArray for dst
    cdef GpuMat dst_gpu = GpuMat()
    warpPerspective(src_gpu, dst_gpu, M_mat, size, INTER_NEAREST)

    # Get result of dst
    cdef Mat dst_host
    dst_gpu.download(dst_host)
    cdef np.ndarray out = <np.ndarray> pyopencv_from(dst_host)
    return out

def warpPerspectiveWrapper(np.ndarray[np.uint8_t, ndim=2] _src,
                               np.ndarray[np.float32_t, ndim=2] _M,
                               _size_tuple,
                               int _flags=INTER_NEAREST):
    # Create GPU/device InputArray for src
    cdef Mat src_mat = Mat()
    pyopencv_to(<PyObject*> _src, src_mat)

    # Create CPU/host InputArray for M
    cdef Mat M_mat = Mat()
    pyopencv_to(<PyObject*> _M, M_mat)

    # Create Size object from size tuple
    # Note that size/shape in Python is handled in row-major-order -- therefore, width is [1] and height is [0]
    cdef Size size = Size(<int> _size_tuple[1], <int> _size_tuple[0])

    # Create empty GPU/device OutputArray for dst
    cdef Mat dst_mat = Mat()
    warpPerspective(src_mat, dst_mat, M_mat, size, INTER_NEAREST)

    # Get result of dst
    cdef np.ndarray out = <np.ndarray> pyopencv_from(dst_mat)
    return out

def echoPyObject(object arr):
    cdef Mat M
    pyopencv_to(<PyObject*> arr, M)
    return <object> pyopencv_from(M)

def cudaCvtColorGray(object img):
    cdef Mat m
    cdef GpuMat m_gpu
    cdef GpuMat out_gpu
    pyopencv_to(<PyObject*> img, m)
    m_gpu.upload(m)
    cvtColor(m_gpu, out_gpu, COLOR_BGR2GRAY)
    out_gpu.download(m)
    return <object> pyopencv_from(m)

def init():
    np.import_array()