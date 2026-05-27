""" Max value projection of a 3D volume onto a 2D plane. """

import numpy as np
import numpy.typing as npt
from typing import Optional, Sequence

from .project3dC import max_value3C

DEFAULT_MIN_VALUE = 0.0


def maximize(
    input_volume: npt.ArrayLike,
    matrix: npt.ArrayLike,
    shape: Optional[Sequence[int]] = None,
    min_value: float = DEFAULT_MIN_VALUE,
) -> np.ndarray:
    """Project a 3D volume onto a 2D plane using the maximum value.

    Args:
        input_volume: The 3D volume to project.
        matrix: The projection matrix.
        shape: The shape of the output 2D image.
        min_value: The minimum value for the output image; values at or below are ignored.

    Returns:
        The projected 2D image.
    """
    input_volume = np.ascontiguousarray(input_volume)
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    if shape is None:
        shape = input_volume.shape[:2]
    output_plane = np.empty(shape, dtype=input_volume.dtype, order="C")
    max_value3C(input_volume, output_plane, matrix, min_value)
    return output_plane


class Maximizer:
    """A class for maximizing a 3D volume onto a 2D plane."""

    def __init__(
        self,
        input_volume: npt.ArrayLike,
        shape: Optional[Sequence[int]] = None,
        min_value: float = DEFAULT_MIN_VALUE,
    ) -> None:
        """Initialize the Maximizer.

        Args:
            input_volume: The 3D volume to project.
            shape: The shape of the output 2D image.
            min_value: The minimum value for the output image; values at or below are ignored.
        """
        self.input_volume = np.ascontiguousarray(input_volume)
        self.min_value = min_value
        if shape is None:
            shape = self.input_volume.shape[:2]
        self.output_plane = np.empty(shape, dtype=self.input_volume.dtype, order="C")

    def maximize(self, matrix: npt.ArrayLike) -> np.ndarray:
        """Maximize the 3D volume onto the preallocated output array.

        Args:
            matrix: The projection matrix.

        Returns:
            The projected 2D image.
        """
        matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        max_value3C(self.input_volume, self.output_plane, matrix, self.min_value)
        return self.output_plane
