#!/usr/bin/env python

from setuptools import setup

setup(name='barbados-clouds',
      version='1.0',
      description='Object based cloud detection and classification',
      author='Johanna Roschke',
      author_email='roschke.johanna@web.de',
      url='https://github.com/remsens-lim/barbados-clouds.git',
      license='GPL-3.0',
      packages=['classify'],
      package_dir={"": "src"},
      package_data={"": ["*.json"]},
      include_package_data=True,
      install_requires=[
          'numpy',
          'xarray',
          'scipy',
          'scikit-image',
          'opencv-python',
          'matplotlib',
          'pandas',
          'datetime',
          'openpyxl'

      ],
     )
