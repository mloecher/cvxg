import os
from sys import platform as _platform

if _platform == "darwin":
    os.environ["CC"] = "gcc-5"
    os.environ["CXX"] = "g++-5"

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize
import numpy

sourcefiles = ['cvxg.pyx', '../src/cvx_matrix.c', '../src/op_gradient.c', '../src/op_bval.c', '../src/op_beta.c', '../src/op_slewrate.c', '../src/op_moments.c']

if _platform == "darwin":
    extensions = [Extension("cvxg",
                    sourcefiles,
                    language="c",
                    include_dirs=[".",  "../src", "/usr/local/include/", numpy.get_include()],
                    library_dirs=[".", "../src", "/usr/local/lib/"],
                    extra_compile_args=['-std=c11'],
                   )]
elif _platform == "win32":
    extensions = [Extension("cvxg",
                    sourcefiles,
                    language="c",
                    include_dirs=[".", "../src", numpy.get_include()],
                    library_dirs=[".", "../src"],
                    extra_compile_args=['-std=c11'],
                   )]

setup(
    ext_modules = cythonize(extensions)
)