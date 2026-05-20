import numpy as np
import numpy.typing as npt
from typing import Optional, Sequence

from .resample3C import resample3C


def sample(
    input_volume: npt.ArrayLike,
    matrix: npt.ArrayLike,
    shape: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """To be filled in...

    Args:
        input_volume: To be filled in...
        matrix: To be filled in...
        shape: To be filled in...

    Returns:
        To be filled in...
    """
    input_volume = np.ascontiguousarray(input_volume)
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    if shape is None:
        shape = input_volume.shape
    output_volume = np.empty(shape, dtype=input_volume.dtype, order="C")
    resample3C(input_volume, output_volume, matrix, 0.0)
    return output_volume


class Sampler:
    """To be filled in..."""

    def __init__(self, input_volume: npt.ArrayLike, shape: Optional[Sequence[int]] = None) -> None:
        """To be filled in...

        Args:
            input_volume: To be filled in...
            shape: To be filled in...
        """
        self.input_volume = np.ascontiguousarray(input_volume)
        if shape is None:
            shape = self.input_volume.shape
        self.output_volume = np.empty(shape, dtype=self.input_volume.dtype, order="C")

    def sample(self, matrix: npt.ArrayLike) -> np.ndarray:
        """To be filled in...

        Args:
            matrix: To be filled in...

        Returns:
            To be filled in...
        """
        matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        resample3C(self.input_volume, self.output_volume, matrix, 0.0)
        return self.output_volume
