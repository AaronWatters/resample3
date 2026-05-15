import numpy as np

from .project3dC import max_value3C


def maximize(input_volume, matrix, shape=None, min_value=0.0):
    input_volume = np.ascontiguousarray(input_volume)
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    if shape is None:
        shape = input_volume.shape[:2]
    output_plane = np.empty(shape, dtype=input_volume.dtype, order="C")
    max_value3C(input_volume, output_plane, matrix, min_value)
    return output_plane


class Maximizer:
    """Max-value projection of a fixed input volume.

    Pre-allocates the output plane once so that repeated calls to
    :meth:`maximize` avoid allocation overhead.

    Parameters
    ----------
    input_volume : array-like, shape (D0, D1, D2)
        3-D source volume.  Will be made C-contiguous on construction.
    matrix : array-like, shape (4, 4)
        Input-to-output mapping matrix (float64).  Each input voxel
        coordinate ``(x, y, z, 1)`` is mapped to output pixel coordinates
        via ``matrix @ [x, y, z, 1]``.
    shape : tuple of int, optional
        Shape ``(H, W)`` of the 2-D output plane.  Defaults to
        ``input_volume.shape[:2]``.
    min_value : float, optional
        Background fill value for output pixels that receive no
        projection (default ``0.0``).
    """

    def __init__(self, input_volume, matrix, shape=None, min_value=0.0):
        self.input_volume = np.ascontiguousarray(input_volume)
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        self.min_value = min_value
        if shape is None:
            shape = self.input_volume.shape[:2]
        self.output_plane = np.empty(shape, dtype=self.input_volume.dtype, order="C")

    def maximize(self):
        """Run the max-value projection and return the output plane.

        The same buffer is reused on every call; copy the result before
        calling ``maximize`` again if you need to retain previous output.
        """
        max_value3C(self.input_volume, self.output_plane, self.matrix, self.min_value)
        return self.output_plane
