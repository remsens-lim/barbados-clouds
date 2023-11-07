#!/usr/bin/env python

from setuptools import setup

setup(name='barbados-clouds',
      version='1.0',
      description='Object based cloud detection and classification',
      author='Johanna Roschke',
      author_email='roschke.johanna@web.de',
      url='https://github.com/remsens-lim/barbados-clouds',
      license='MIT',
      packages=[],
      install_requires=[
          'numpy',
          'xarray',
          'scipy',
          'scikit-image',
          'opencv-python',
          'matplotlib',
          'pandas',
          'datetime'

      ],
     )
