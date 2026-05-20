"""To be filled in..."""

import numpy as np
import numpy.typing as npt
from typing import Optional, Sequence

from .matrices import apply_matrix_to_vector
from .slice3dC import slice3dC

DEFAULT_MIN_VALUE = 0.0


def slice3py(
    output_matrix: np.ndarray,
    input_volume: np.ndarray,
    depth: float,
    invmatrix: np.ndarray,
    min_value: float = 0.0,
) -> None:
    """To be filled in...

    Args:
        output_matrix: To be filled in...
        input_volume: To be filled in...
        depth: To be filled in...
        invmatrix: To be filled in...
        min_value: To be filled in...

    Returns:
        To be filled in...
    """
    output_matrix[...] = min_value
    for i in range(output_matrix.shape[0]):
        for j in range(output_matrix.shape[1]):
            x, y, z = apply_matrix_to_vector(invmatrix, (i, j, depth))
            if 0 <= x < input_volume.shape[0] and 0 <= y < input_volume.shape[1] and 0 <= z < input_volume.shape[2]:
                output_matrix[i, j] = max(min_value, input_volume[int(x), int(y), int(z)])


def slicepy(
    input_volume: np.ndarray,
    depth: float,
    invmatrix: np.ndarray,
    shape: Optional[Sequence[int]] = None,
    min_value: float = 0.0,
) -> np.ndarray:
    """To be filled in...

    Args:
        input_volume: To be filled in...
        depth: To be filled in...
        invmatrix: To be filled in...
        shape: To be filled in...
        min_value: To be filled in...

    Returns:
        To be filled in...
    """
    if shape is None:
        shape = input_volume.shape[:2]
    output_matrix = np.empty(shape, dtype=input_volume.dtype, order="C")
    slice3py(output_matrix, input_volume, depth, invmatrix, min_value)
    return output_matrix


def slice(
    input_volume: npt.ArrayLike,
    depth: float,
    invmatrix: npt.ArrayLike,
    shape: Optional[Sequence[int]] = None,
    min_value: float = DEFAULT_MIN_VALUE,
    output_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """To be filled in...

    Args:
        input_volume: To be filled in...
        depth: To be filled in...
        invmatrix: To be filled in...
        shape: To be filled in...
        min_value: To be filled in...
        output_matrix: To be filled in...

    Returns:
        To be filled in...
    """
    input_volume = np.ascontiguousarray(input_volume)
    invmatrix = np.ascontiguousarray(invmatrix, dtype=np.float64)
    if output_matrix is None:
        if shape is None:
            shape = input_volume.shape[:2]
        output_matrix = np.empty(shape, dtype=input_volume.dtype, order="C")
    elif shape is not None and tuple(shape) != output_matrix.shape:
        raise ValueError("shape must match output_matrix shape")
    slice3dC(output_matrix, input_volume, depth, invmatrix, min_value)
    return output_matrix


class Slicer:
    """To be filled in..."""

    def __init__(
        self,
        input_volume: npt.ArrayLike,
        shape: Optional[Sequence[int]] = None,
        min_value: float = DEFAULT_MIN_VALUE,
    ) -> None:
        """To be filled in...

        Args:
            input_volume: To be filled in...
            shape: To be filled in...
            min_value: To be filled in...
        """
        self.input_volume = np.ascontiguousarray(input_volume)
        self.min_value = min_value
        if shape is None:
            shape = self.input_volume.shape[:2]
        self.output_matrix = np.empty(shape, dtype=self.input_volume.dtype, order="C")

    def slice(
        self,
        invmatrix: npt.ArrayLike,
        depth: float,
        min_value: Optional[float] = None,
    ) -> np.ndarray:
        """To be filled in...

        Args:
            invmatrix: To be filled in...
            depth: To be filled in...
            min_value: To be filled in...

        Returns:
            To be filled in...
        """
        if min_value is None:
            min_value = self.min_value
        slice(
            self.input_volume,
            depth,
            invmatrix,
            shape=self.output_matrix.shape,
            min_value=min_value,
            output_matrix=self.output_matrix,
        )
        return self.output_matrix
