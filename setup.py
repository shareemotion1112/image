# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 18:39:29 2021

@author: coolc
"""

from distutils.core import setup
# from distutils.extension import Extension


# from setuptools import setup
from Cython.Build import cythonize
import numpy




extensions = ['E:/COOLCSG/Python Projects/image/polygonFit/polygonFitCython.pyx']

setup(
    ext_modules=cythonize(extensions),
    include_dirs = [numpy.get_include()]
)
 



   