import os
import numpy as np
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

"""
References
* http://cython.readthedocs.io/en/latest/src/reference/compilation.html
"""

CUR_DIR = os.path.dirname(__file__)

extensions = [
    Extension('GpuWrapper',
              sources=[os.path.join(CUR_DIR, 'GpuWrapper.pyx')],
              language='c++')
]

setup(
    cmdclass={'build_ext': build_ext},
    name="GpuWrapper",
    ext_modules=cythonize(extensions, include_path=[np.get_include()])
)
