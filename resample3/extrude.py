import numpy as np
import numpy.typing as npt
from typing import Optional, Sequence, Tuple

from .project3dC import extrude3C


def extrude(
    input_volume: npt.ArrayLike,
    matrix: npt.ArrayLike,
    shape: Optional[Sequence[int]] = None,
    min_value: float = 0.0,
    max_depth: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrude a 3D volume into a 2D image.

    Args:
        input_volume: The 3D volume to extrude.
        matrix: The projection matrix.
        shape: The shape of the output 2D image.
        min_value: The minimum value for the output image.
        max_depth: The maximum depth for the output image.

    Returns:
        A tuple containing the extruded 2D image and the corresponding depth values.
    """
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
    """A class for extruding 3D volumes into 2D images."""

    def __init__(
        self,
        input_volume: npt.ArrayLike,
        shape: Optional[Sequence[int]] = None,
        min_value: float = 0.0,
        max_depth: Optional[int] = None,
    ) -> None:
        """Initialize the Extruder.

        Args:
            input_volume: The 3D volume to extrude.
            shape: The shape of the output 2D image.
            min_value: The minimum value for the output image; values at or below are ignored.
            max_depth: The maximum depth for the output image.
        """
        self.input_volume = np.ascontiguousarray(input_volume)
        self.min_value = min_value
        if max_depth is None:
            max_depth = 2 * max(self.input_volume.shape)
        self.max_depth = max_depth
        if shape is None:
            shape = self.input_volume.shape[:2]
        self.output_plane = np.empty(shape, dtype=self.input_volume.dtype, order="C")
        self.output_depths = np.empty(shape, dtype=np.float64, order="C")

    def extrude(
        self,
        matrix: npt.ArrayLike,
        min_value: Optional[float] = None,
        max_depth: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extrude the 3D volume into the preallocated output arrays.

        Args:
            matrix: The projection matrix.
            min_value: The minimum value for the output image; values at or below are ignored.
            max_depth: The maximum depth for the output image.

        Returns:
            A tuple containing the extruded 2D image and the corresponding depth values.
        """
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
