import numpy as np

from .project3dC import max_value3C


def maximize(input_volume, matrix, shape=None):
    input_volume = np.ascontiguousarray(input_volume)
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    if shape is None:
        shape = input_volume.shape[:2]
    output_plane = np.empty(shape, dtype=input_volume.dtype, order="C")
    max_value3C(input_volume, output_plane, matrix, 0.0)
    return output_plane


class Maximizer:
    def __init__(self, input_volume, matrix, shape=None):
        self.input_volume = np.ascontiguousarray(input_volume)
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        if shape is None:
            shape = self.input_volume.shape[:2]
        self.output_plane = np.empty(shape, dtype=self.input_volume.dtype, order="C")

    def maximize(self, matrix=None):
        if matrix is None:
            matrix = self.matrix
        else:
            matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        max_value3C(self.input_volume, self.output_plane, matrix, 0.0)
        return self.output_plane
