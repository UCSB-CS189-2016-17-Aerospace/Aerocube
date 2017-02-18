import numpy as np  # Import Python functions, attributes, submodules of numpy
cimport numpy as np  # Import C functions, attributes, submodules of numpy
from libcpp.string cimport string  # Import <string>
from libc.string cimport memcpy  # Import <string.h>, or <cstring>

"""
References:
* http://makerwannabe.blogspot.com/2013/09/calling-opencv-functions-via-cython.html
"""

# Define CV_8UC1 (8-bit, 1-channel matrix/image from core.hpp)
cdef extern from 'opencv2/core/core.hpp':
    cdef int CV_8UC1

# Define C++ class Mat from core.hpp
cdef extern from 'opencv2/core/core.hpp' namespace 'cv':
    cdef cppclass Mat:
        # Constructor
        Mat() except +
        # Enum member
        int AUTO_STEP
        void create(int, int, int)
        # void create(int, int, int, void*, size_t) except +
        void* data

cdef void grayArr2cvMat(np.ndarray arr, Mat& out):
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
    # out.create(<int> rows, <int> cols, CV_8UC1, <void *> im_buff, <size_t> out.AUTO_STEP)
    out.create(rows, cols, CV_8UC1)
    memcpy(out.data, im_buff, rows*cols)

