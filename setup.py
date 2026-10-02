from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension("c_bifood", ["c_bifood.pyx"], include_dirs=[np.get_include()], language="c++")
setup(name="c_bifood", ext_modules=cythonize([ext], language_level=3))
