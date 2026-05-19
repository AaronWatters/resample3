"""
Slice a 3D volume into 2D planes using an affine transformation matrix to define the slicing plane.
"""

import numpy as np
from .matrices import apply_matrix_to_vector

def slicepy(input_volume, depth, invmatrix, shape=None, min_value=0.0):
    """naive python implementation of slicing a 3D volume using an inverse transformation matrix."""
    if shape is None:
        shape = input_volume.shape[:2]
    # initialize output to be all min_value
    output = np.full(shape, min_value, dtype=input_volume.dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            # apply the inverse matrix to the (i, j) coordinate to get the corresponding 3D coordinate
            x, y, z = apply_matrix_to_vector(invmatrix, (i, j, depth))
            if 0 <= x < input_volume.shape[0] and 0 <= y < input_volume.shape[1] and 0 <= z < input_volume.shape[2]:
                output[i, j] = max(min_value, input_volume[int(x), int(y), int(z)])
    return output
