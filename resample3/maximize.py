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
    input_volume = np.ascontiguousarray(input_volume)
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    if shape is None:
        shape = input_volume.shape[:2]
    output_plane = np.empty(shape, dtype=input_volume.dtype, order="C")
    max_value3C(input_volume, output_plane, matrix, min_value)
    return output_plane


class Maximizer:
    def __init__(
        self,
        input_volume: npt.ArrayLike,
        shape: Optional[Sequence[int]] = None,
        min_value: float = DEFAULT_MIN_VALUE,
    ) -> None:
        self.input_volume = np.ascontiguousarray(input_volume)
        self.min_value = min_value
        if shape is None:
            shape = self.input_volume.shape[:2]
        self.output_plane = np.empty(shape, dtype=self.input_volume.dtype, order="C")

    def maximize(self, matrix: npt.ArrayLike) -> np.ndarray:
        matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        max_value3C(self.input_volume, self.output_plane, matrix, self.min_value)
        return self.output_plane
