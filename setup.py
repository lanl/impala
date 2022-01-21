import setuptools
from setuptools import setup

setup(name='impala',
      version='0.1',
      description='Bayesian model calibration',
      url='http://www.github.com/lanl/impala',
      author='Devin Francom, others',
      author_email='',
      license='BSD-3',
      packages=setuptools.find_packages(),
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=[
          'numpy',
          'matplotlib',
          'scipy',
          'pyBASS',
          'multiprocessing'
      ]
      )