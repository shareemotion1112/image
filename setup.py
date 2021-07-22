# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 18:39:29 2021

@author: coolc
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('polygonFitCython.pyx')
)
    