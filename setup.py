import numpy as np
from setuptools import Extension, setup

# isort: split

from Cython.Build import cythonize

extensions = [
    Extension(
        "*",
        ["wavelet/*.pyx"],
        # to cimport numpy
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        annotate=True,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
    ),
)
