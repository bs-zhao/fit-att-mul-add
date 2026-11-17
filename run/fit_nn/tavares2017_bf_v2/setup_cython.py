"""Build the optional full-timeline extension without touching c_bifood."""
from pathlib import Path
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

HERE = Path(__file__).resolve().parent
ext = Extension("c_tavares_fulltimeline",
                [str(HERE / "c_tavares_fulltimeline.pyx")],
                include_dirs=[np.get_include()], language="c++")
setup(name="c-tavares-fulltimeline",version="0.1.0",
      ext_modules=cythonize([ext],language_level=3))
