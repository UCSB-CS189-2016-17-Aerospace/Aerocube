import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libc.string import memcpy

"""
References:
* http://makerwannabe.blogspot.com/2013/09/calling-opencv-functions-via-cython.html
"""

cdef extern from 'opencv2/core/core.hpp':
    cdef int CV_8UC1

cdef extern from 'opencv2/core/core.hpp' namespace 'cv':
    cdef cppclass Mat:
        Mat() except +
        void create(int, int, int)
        void create(int, int, int, void*)
        void* data

cdef void grayArr2cvMat(np.ndarray arr, Mat& out):
    assert(arr.ndim == 2, "ASSERT::1 Channel Gray Image Only!")
    cdef np.ndarray[np.uint8_t, ndim=2] np_buff = np.ascontiguousarray(arr, dtype=np.uint8_t)
    cdef unsigned int* im_buff = <unsigned int*> np_buff.data
    cdef int rows = arr.shape[0]
    cdef int cols = arr.shape[1]
    out.create(rows, cols, CV_8UC1, im_buff)
    # out.create(rows, cols, CV_8UC1)
    # memcpy(out.data, im_buff, rows*cols)

