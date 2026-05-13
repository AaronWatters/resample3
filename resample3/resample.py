import numpy as np

from .resample3C import resample3C


def sample(input_volume, matrix, shape=None):
    input_volume = np.ascontiguousarray(input_volume)
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    if shape is None:
        shape = input_volume.shape
    output_volume = np.empty(shape, dtype=input_volume.dtype, order="C")
    resample3C(input_volume, output_volume, matrix, 0.0)
    return output_volume

