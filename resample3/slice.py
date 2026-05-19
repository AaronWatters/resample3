"""
Slice a 3D volume into 2D planes using an affine transformation matrix to define the slicing plane.
"""

import numpy as np

from .matrices import apply_matrix_to_vector


def slice3py(output_matrix, input_volume, depth, invmatrix, min_value=0.0):
    """Naive Python implementation of slicing into a preallocated 2D output matrix."""
    output_matrix[...] = min_value
    for i in range(output_matrix.shape[0]):
        for j in range(output_matrix.shape[1]):
            x, y, z = apply_matrix_to_vector(invmatrix, (i, j, depth))
            if 0 <= x < input_volume.shape[0] and 0 <= y < input_volume.shape[1] and 0 <= z < input_volume.shape[2]:
                output_matrix[i, j] = max(min_value, input_volume[int(x), int(y), int(z)])


def slicepy(input_volume, depth, invmatrix, shape=None, min_value=0.0):
    """Naive Python implementation of slicing a 3D volume using an inverse transformation matrix."""
    if shape is None:
        shape = input_volume.shape[:2]
    output_matrix = np.empty(shape, dtype=input_volume.dtype, order="C")
    slice3py(output_matrix, input_volume, depth, invmatrix, min_value)
    return output_matrix
