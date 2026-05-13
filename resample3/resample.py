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


class Sampler:
    def __init__(self, input_volume, shape=None):
        self.input_volume = np.ascontiguousarray(input_volume)
        if shape is None:
            shape = self.input_volume.shape
        self.output_volume = np.empty(shape, dtype=self.input_volume.dtype, order="C")

    def sample(self, matrix):
        matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        resample3C(self.input_volume, self.output_volume, matrix, 0.0)
        return self.output_volume

