import numpy as np

from .project3dC import extrude3C


def extrude(input_volume, matrix, shape=None, min_value=0.0, max_depth=None):
    input_volume = np.ascontiguousarray(input_volume)
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    if shape is None:
        shape = input_volume.shape[:2]
    if max_depth is None:
        max_depth = 2 * max(input_volume.shape)
    output_plane = np.empty(shape, dtype=input_volume.dtype, order="C")
    output_depths = np.empty(shape, dtype=np.float64, order="C")
    extrude3C(output_plane, output_depths, input_volume, matrix, min_value, max_depth)
    return output_plane, output_depths


class Extruder:
    def __init__(self, input_volume, shape=None, min_value=0.0, max_depth=None):
        self.input_volume = np.ascontiguousarray(input_volume)
        self.min_value = min_value
        if max_depth is None:
            max_depth = 2 * max(self.input_volume.shape)
        self.max_depth = max_depth
        if shape is None:
            shape = self.input_volume.shape[:2]
        self.output_plane = np.empty(shape, dtype=self.input_volume.dtype, order="C")
        self.output_depths = np.empty(shape, dtype=np.float64, order="C")

    def extrude(self, matrix, min_value=None, max_depth=None):
        if min_value is None:
            min_value = self.min_value
        if max_depth is None:
            max_depth = self.max_depth
        matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        extrude3C(
            self.output_plane,
            self.output_depths,
            self.input_volume,
            matrix,
            min_value,
            max_depth,
        )
        return self.output_plane, self.output_depths
