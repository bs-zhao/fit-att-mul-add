from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

HERE = Path(__file__).resolve().parent

ext = Extension(
    "c_bifood",
    [str(HERE / "c_bifood.pyx")],
    include_dirs=[np.get_include()],
    language="c++",
)

setup(
    name="c-bifood",
    version="0.1.0",
    ext_modules=cythonize([ext], language_level=3),
)
