from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize('physical_models_c.pyx', annotate = True),
    include_dirs = [numpy.get_include()],
    )
setup(
    ext_modules = cythonize('pointcloud.pyx', annotate = True),
    include_dirs = [numpy.get_include()],
    )

# EOF
