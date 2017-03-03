from libcpp cimport bool
from cpython.ref cimport PyObject

cdef extern from 'cuda/pyopencv_converter.cpp':
    cdef PyObject* pyopencv_from(const Mat&m)
    cdef bool pyopencv_to(PyObject* o, Mat&m)

cdef extern from 'opencv2/core/core.hpp':
    cdef int CV_8UC1

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


cdef extern from 'opencv2/core/core.hpp' namespace 'cv':
    cdef cppclass Mat:
        Mat() except +
        void create(int, int, int) except +
        void* data
        int rows
        int cols
    cdef void meanStdDev(Mat src, Mat mean, Mat stddev)

cdef extern from 'opencv2/core/cuda.hpp' namespace 'cv::cuda':
    cdef cppclass GpuMat:
        GpuMat() except +
        void upload(Mat arr) except +
        void download(Mat dst) const
        GpuMat colRange(int startcol, int endcol) const
        GpuMat rowRange(int startrow, int endrow) const
    cdef cppclass Stream:
        Stream() except +

cdef extern from 'opencv2/cudawarping.hpp' namespace 'cv::cuda':
    cdef void warpPerspective(GpuMat src, GpuMat dst, Mat M, Size dsize, int flags)
