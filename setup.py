#!/usr/bin/env python
# encoding: utf-8
"""
this file has setup to distribute this as a package
"""

from setuptools import setup

setup(
        name = "steme",
        version = 0.1,
        description = "",
        url = "",
        author = "Giovana Morais",
        author_email = "giovana.vmorais@gmail.com",
        license = "",
        packages = ["steme"],
        install_requires = [
            "h5py>=3.7",
            "librosa>=0.8.0",
            "mirdata>=0.3.7",
            "numpy>=1.19.2",
            "pandas>=2.0.0",
            "scipy>=1.9.0",
            "tensorflow>=2.0"
        ],
        zip_safe = False
)
