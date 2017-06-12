try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
  name = 'cuturi algos',
  ext_modules = cythonize("algos.pyx"),
  include_dirs=[np.get_include()]
)
