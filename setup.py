from setuptools import Extension, find_packages, setup
import numpy


setup(
    name="resample3",
    version="0.0.0",
    packages=find_packages(),
    ext_modules=[
        Extension(
            "resample3.resample3C",
            ["resample3dC.c"],
            include_dirs=[numpy.get_include()],
        ),
        Extension(
            "resample3.project3dC",
            ["project3dC.c"],
            include_dirs=[numpy.get_include()],
        ),
    ],
)
