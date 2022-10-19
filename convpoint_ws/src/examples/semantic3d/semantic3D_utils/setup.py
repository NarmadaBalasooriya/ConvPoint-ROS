from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension(
       "semantic3D",
       sources=["Semantic3D.pyx", "Sem3D.cxx",],  # source file(s)
       include_dirs=["./", numpy.get_include()],
       language="c++",            
       extra_compile_args = [ "-std=c++11", "-fopenmp",],
       extra_link_args=["-std=c++11", '-fopenmp'],
  )]

setup(
    name = "Semantic3D_utils",
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext},
)
