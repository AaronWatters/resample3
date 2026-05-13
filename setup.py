from setuptools import Extension, setup
import numpy


setup(
    name="resample3dC",
    version="0.0.0",
    ext_modules=[
        Extension(
            "resample3dC",
            ["resample3dC.c"],
            include_dirs=[numpy.get_include()],
        )
    ],
)
