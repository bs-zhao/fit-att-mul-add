from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "fit_att_mul_add.simulators.c_bifood",
        ["src/fit_att_mul_add/simulators/c_bifood.pyx"],
        language="c++",
        include_dirs=[np.get_include()],
    )
]

setup(
    name="fit-att-mul-add-simulator",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    package_dir={"": "src"},
)
