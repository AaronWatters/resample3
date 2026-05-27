"""To be filled in..."""

import numpy as np
from typing import Optional, Sequence

def apply_matrix_to_vector(matrix: np.ndarray, vector: Sequence[float]) -> np.ndarray:
    """Apply a 4x4 affine transformation matrix to a 3D vector.

    Args:
        matrix: 4 x 4 affine transformation matrix.
        vector: 3D vector to transform.

    Returns:
        Transformed 3D vector.
    """
    if matrix.shape != (4, 4):
        raise ValueError(f"Matrix must be 4x4, got shape {matrix.shape}")
    if len(vector) != 3:
        raise ValueError(f"Vector must have 3 elements, got length {len(vector)}")
    
    # Convert the input vector to homogeneous coordinates (x, y, z, w)
    homogeneous_vector = np.array([vector[0], vector[1], vector[2], 1.0], dtype=np.float64)
    
    # Apply the transformation
    transformed_homogeneous = matrix @ homogeneous_vector
    
    # Convert back to Cartesian coordinates (x', y', z')
    w = transformed_homogeneous[3]
    if w == 0:
        raise ValueError("Transformation resulted in w=0, cannot convert back to Cartesian coordinates")
    
    transformed_vector = transformed_homogeneous[:3] / w
    return transformed_vector

def scale_matrix(sx: float, sy: Optional[float] = None, sz: Optional[float] = None) -> np.ndarray:
    """Create a 4x4 scaling matrix.

    Args:
        sx: Scale factor along the x-axis.
        sy: Scale factor along the y-axis.
        sz: Scale factor along the z-axis.

    Returns:
        4x4 scaling matrix.
    """
    if sy is None:
        sy = sx
    if sz is None:
        sz = sx
    return np.array(
        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

def translation_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    """Create a 4x4 translation matrix.

    Args:
        tx: Translation along the x-axis.
        ty: Translation along the y-axis.
        tz: Translation along the z-axis.

    Returns:
        4x4 translation matrix.
    """
    return np.array(
        [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

def rotation_matrix_z(angle_radians: float) -> np.ndarray:
    """Create a 4x4 rotation matrix around the z-axis.

    Args:
        angle_radians: Rotation angle in radians.

    Returns:
        4x4 rotation matrix.
    """
    #angle_radians = np.radians(angle_degrees)
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)
    return np.array(
        [
            [cos_a, -sin_a, 0.0, 0.0],
            [sin_a, cos_a, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

def rotation_matrix_y(angle_radians: float) -> np.ndarray:
    """Create a 4x4 rotation matrix around the y-axis.

    Args:
        angle_radians: Rotation angle in radians.

    Returns:
        4x4 rotation matrix.
    """
    #angle_radians = np.radians(angle_degrees)
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)
    return np.array(
        [
            [cos_a, 0.0, sin_a, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin_a, 0.0, cos_a, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )   

def rotation_matrix_x(angle_radians: float) -> np.ndarray:
    """Create a 4x4 rotation matrix around the x-axis.

    Args:
        angle_radians: Rotation angle in radians.

    Returns:
        4x4 rotation matrix.
    """
    #angle_radians = np.radians(angle_degrees)
    cos_a = np.cos(angle_radians)
    sin_a = np.sin(angle_radians)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_a, -sin_a, 0.0],
            [0.0, sin_a, cos_a, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

# The diameter of a sphere containing a cube with sides length of 1.
CUBE_DIAMETER = np.sqrt(3)
DEFAULT_SCALING = 1/CUBE_DIAMETER

def projection_matrix(
    from_shape3d: Sequence[float],
    to_shape2d: Sequence[float],
    scales: Optional[Sequence[float]] = None,
    rx: float = 0.0,
    ry: float = 0.0,
    rz: float = 0.0,
) -> np.ndarray:
    """Create a projection matrix from a 3D shape to a 2D shape.

    Args:
        from_shape3d: Source 3D shape (depth, height, width).
        to_shape2d: Target 2D shape (height, width).
        scales: Scale factors for each axis.
        rx: Rotation around the x-axis in radians.
        ry: Rotation around the y-axis in radians.
        rz: Rotation around the z-axis in radians.

    Returns:
        4x4 projection matrix.
    """
    from_shape3d = np.asarray(from_shape3d, dtype=np.float64)
    to_shape2d = np.asarray(to_shape2d, dtype=np.float64)
    if from_shape3d.shape != (3,):
        raise ValueError(f"from_shape3d must be a 3-element shape tuple, got {from_shape3d}")
    if to_shape2d.shape != (2,):
        raise ValueError(f"to_shape2d must be a 2-element shape tuple, got {to_shape2d}")
    scale_ratio = min(to_shape2d / from_shape3d[:2])
    if scales is None:
        scale = scale_matrix(DEFAULT_SCALING * scale_ratio)
    else:
        scales = np.asarray(scales, dtype=np.float64)
        scale = scale_matrix(*(scales * scale_ratio))
    to_center2d = translation_matrix(to_shape2d[0] / 2, to_shape2d[1] / 2, 0)
    #scaled_from_shape3d = apply_matrix_to_vector(scale, from_shape3d)
    to_origin3d = translation_matrix(*(-from_shape3d/2))
    rotate_x = rotation_matrix_x(rx)
    rotate_y = rotation_matrix_y(ry)
    rotate_z = rotation_matrix_z(rz)
    result = to_center2d @ rotate_x @ rotate_y @ rotate_z @ scale @ to_origin3d
    return result
