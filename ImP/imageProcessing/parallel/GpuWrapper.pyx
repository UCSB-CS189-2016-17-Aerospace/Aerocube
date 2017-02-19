import numpy as np  # Import Python functions, attributes, submodules of numpy
cimport numpy as np  # Import C functions, attributes, submodules of numpy
from libcpp.string cimport string  # Import <string>
from libcpp cimport bool  # Get support for C++ bool
from libc.string cimport memcpy  # Import <string.h>, or <cstring>
from cpython.ref cimport PyObject

"""
References:
* http://makerwannabe.blogspot.com/2013/09/calling-opencv-functions-via-cython.html
"""

cdef extern from '/usr/local/lib/opencv/modules/python/src2/pycompat.hpp' namespace 'cv':
    cdef PyObject* pyopencv_from(const Mat& m) except +
    cdef bool pyopencv_to(PyObject* o, Mat& m) except +

cdef extern from 'opencv2/imgproc.hpp' namespace 'cv':
    cdef enum InterpolationFlags:
        INTER_NEAREST = 0

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

cdef void intArr2cvMat(np.ndarray[np.uint8_t, ndim=2] arr, Mat& out):
    """
    Convert a Python numpy object to a C++ Mat object
    :param arr: Python numpy object of types np.ndarray; must be a 1 channel (grayscale) image
    :param out: C++ Mat object for output
    :return: void
    """
    assert(arr.ndim == 2, "ASSERT::1 Channel Gray Image Only!")
    # Note that we are using np.uint8_t (C++ object) for np_buff, but np.uint8 (Python object) in the Python method call
    cdef np.ndarray[np.uint8_t, ndim=2] np_buff = np.ascontiguousarray(arr, dtype=np.uint8)
    cdef unsigned int* im_buff = <unsigned int*> np_buff.data
    cdef int rows = arr.shape[0]
    cdef int cols = arr.shape[1]
    out.create(rows, cols, CV_8UC1)
    memcpy(out.data, im_buff, rows*cols)

cdef void float32Arr2cvMat(np.ndarray[np.float32_t, ndim=2] arr, Mat& out):
    assert(arr.ndim ==2)
    cdef np.ndarray[np.float32_t, ndim=2] np_buff = np.ascontiguousarray(arr, dtype=np.float32)
    cdef float* im_buff = <float*> np_buff.data
    cdef int rows = arr.shape[0]
    cdef int cols = arr.shape[1]
    out.create(rows, cols, CV_32FC1)
    memcpy(out.data, im_buff, rows*cols)

cdef void intArr2cvGpuMat(np.ndarray[np.uint8_t, ndim=2] arr, GpuMat& out):
    cdef Mat mat
    intArr2cvMat(arr, mat)
    out.upload(mat)

cdef np.ndarray[np.uint8_t, ndim=2] cvMat2IntArr(Mat& m):
    cdef np.ndarray[np.uint8_t, ndim=2] np_ret = np.zeros((m.rows, m.cols), dtype=np.uint8, order='C')
    print(np_ret.flags)
    cdef unsigned int* im_buff = <unsigned int*> np_ret.data
    memcpy(im_buff, m.data, m.rows * m.cols)
    return np_ret

# cdef np.ndarray[np.float32_t, ndim=2] cvMat2Float32Arr(Mat& m):
#     cdef np.ndarray[np.float32_t, ndim=2] np_ret = np.zeros((m.rows, m.cols), dtype=np.float32)
#     memcpy(<float*> np_ret.data, <float*> m.data, m.rows * m.cols)
#     return np_ret

def cudaWarpPerspectiveWrapper(np.ndarray[np.uint8_t, ndim=2] _src,
                               np.ndarray[np.float32_t, ndim=2] _M,
                               _size_tuple,
                               int _flags=INTER_NEAREST):
    # Create GPU/device InputArray for src
    cdef GpuMat src_mat = GpuMat()
    intArr2cvGpuMat(_src, src_mat)

    # Create CPU/host InputArray for M
    cdef Mat M_mat = Mat()
    float32Arr2cvMat(_M, M_mat)
    print((<float*>(M_mat.data))[0])
    print((<float*>(M_mat.data))[1])
    print((<float*>(M_mat.data))[2])

    # Create Size object from size tuple
    # Note that size/shape in Python is handled in row-major-order -- therefore, width is [1] and height is [0]
    cdef Size size = Size(<int> _size_tuple[1], <int> _size_tuple[0])

    # Create empty GPU/device OutputArray for dst
    cdef GpuMat dst_gpu = GpuMat()
    warpPerspective(src_mat, dst_gpu, M_mat, size, INTER_NEAREST)
    # cdef Mat tmp
    # intArr2cvMat(_src, tmp)
    # dst_gpu.upload(tmp)

    # Get result of dst
    cdef Mat dst_host
    dst_gpu.download(dst_host)
    cdef np.ndarray out = cvMat2IntArr(dst_host)
    return out

def warpPerspectiveWrapper(np.ndarray[np.uint8_t, ndim=2] _src,
                               np.ndarray[np.float32_t, ndim=2] _M,
                               _size_tuple,
                               int _flags=INTER_NEAREST):
    # Create GPU/device InputArray for src
    cdef Mat src_mat = Mat()
    intArr2cvMat(_src, src_mat)

    # Create CPU/host InputArray for M
    cdef Mat M_mat = Mat()
    float32Arr2cvMat(_M, M_mat)
    print((<float*>(M_mat.data))[0])
    print((<float*>(M_mat.data))[1])
    print((<float*>(M_mat.data))[2])

    # Create Size object from size tuple
    # Note that size/shape in Python is handled in row-major-order -- therefore, width is [1] and height is [0]
    cdef Size size = Size(<int> _size_tuple[1], <int> _size_tuple[0])

    # Create empty GPU/device OutputArray for dst
    cdef Mat dst_mat = Mat()
    warpPerspective(src_mat, dst_mat, M_mat, size, INTER_NEAREST)

    # Get result of dst
    cdef np.ndarray out = cvMat2IntArr(dst_mat)
    return out

def echoPyObject(PyObject* arr):
    np.import_array()
    cdef Mat M
    pyopencv_to(arr, M)
    return pyopencv_from(M)
