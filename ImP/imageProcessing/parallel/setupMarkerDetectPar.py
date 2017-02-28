import os
import numpy as np
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

# Determine current directory of this setup file to find our module
CUR_DIR = os.path.dirname(__file__)

extensions = [
    Extension('markerDetectPar',
              sources=[os.path.join(CUR_DIR, 'markerDetectPar.pyx')],
              language='c++',
              include_dirs=[np.get_include()])
]

setup(
    cmdclass={'build_ext': build_ext},
    name='markerDetectPar',
    ext_modules=cythonize(extensions)
)