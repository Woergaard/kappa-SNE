from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "_utils", 
        ["_utils.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "_barnes_hut_ksne", 
        ["_barnes_hut_ksne.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    ext_modules = cythonize(extensions)
)