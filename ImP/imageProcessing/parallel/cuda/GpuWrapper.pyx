import numpy as np  # Import Python functions, attributes, submodules of numpy
cimport numpy as np  # Import C functions, attributes, submodules of numpy

"""
Implementation file for GpuWrapper
Definitions can be found in GpuWrapper.pxd
References:
* http://makerwannabe.blogspot.com/2013/09/calling-opencv-functions-via-cuda.html
* On converting PyObjects to OpenCV Mat objects
    * http://stackoverflow.com/questions/22736593/what-is-the-easiest-way-to-convert-ndarray-into-cvmat
    * http://stackoverflow.com/questions/12957492/writing-python-bindings-for-c-code-that-use-opencv/12972689#12972689
    * http://stackoverflow.com/questions/13745265/exposing-opencv-based-c-function-with-mat-numpy-conversion-to-python
    * https://github.com/Algomorph/pyboostcvconverter/blob/master/src/pyboost_cv3_converter.cpp
"""

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
    warpPerspective(src_gpu, dst_gpu, M_mat, size, _flags)

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
    warpPerspective(src_mat, dst_mat, M_mat, size, _flags)

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
    """
    Calls numpy's import_array() function, which *must* be done before any PyObject conversions happen.
    This function *must* be called immediately after GpuWrapper is imported!
    :return:
    """
    np.import_array()