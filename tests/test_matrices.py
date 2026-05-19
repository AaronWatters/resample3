import numpy as np
import pytest

from resample3.matrices import (
    apply_matrix_to_vector,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    scale_matrix,
    translation_matrix,
)


def test_apply_matrix_to_vector_identity_returns_same_vector():
    matrix = np.eye(4, dtype=np.float64)
    vector = np.array([1.5, -2.0, 3.25], dtype=np.float64)

    result = apply_matrix_to_vector(matrix, vector)

    assert result.shape == (3,)
    assert result.dtype == np.float64
    assert np.allclose(result, vector, atol=1e-12)


def test_apply_matrix_to_vector_translation_applies_offset():
    matrix = translation_matrix(2.0, -3.0, 5.0)
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    result = apply_matrix_to_vector(matrix, vector)

    assert np.allclose(result, np.array([3.0, -1.0, 8.0]), atol=1e-12)


def test_apply_matrix_to_vector_scale_applies_axis_factors():
    matrix = scale_matrix(2.0, 3.0, 4.0)
    vector = np.array([1.0, -2.0, 0.5], dtype=np.float64)

    result = apply_matrix_to_vector(matrix, vector)

    assert np.allclose(result, np.array([2.0, -6.0, 2.0]), atol=1e-12)


@pytest.mark.parametrize("angle", [-np.pi / 2, 0.0, np.pi / 2])
def test_apply_matrix_to_vector_matches_rotation_x_matrix_multiplication(angle):
    matrix = rotation_matrix_x(angle)
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    result = apply_matrix_to_vector(matrix, vector)
    expected = (matrix @ np.array([1.0, 2.0, 3.0, 1.0], dtype=np.float64))[:3]

    assert np.allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("angle", [-np.pi / 2, 0.0, np.pi / 2])
def test_apply_matrix_to_vector_matches_rotation_y_matrix_multiplication(angle):
    matrix = rotation_matrix_y(angle)
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    result = apply_matrix_to_vector(matrix, vector)
    expected = (matrix @ np.array([1.0, 2.0, 3.0, 1.0], dtype=np.float64))[:3]

    assert np.allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("angle", [-np.pi / 2, 0.0, np.pi / 2])
def test_apply_matrix_to_vector_matches_rotation_z_matrix_multiplication(angle):
    matrix = rotation_matrix_z(angle)
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    result = apply_matrix_to_vector(matrix, vector)
    expected = (matrix @ np.array([1.0, 2.0, 3.0, 1.0], dtype=np.float64))[:3]

    assert np.allclose(result, expected, atol=1e-12)


def test_apply_matrix_to_vector_raises_for_wrong_matrix_shape():
    matrix = np.eye(3, dtype=np.float64)

    with pytest.raises(ValueError):
        apply_matrix_to_vector(matrix, np.array([1.0, 2.0, 3.0], dtype=np.float64))


def test_apply_matrix_to_vector_raises_for_wrong_vector_length():
    matrix = np.eye(4, dtype=np.float64)

    with pytest.raises(ValueError):
        apply_matrix_to_vector(matrix, np.array([1.0, 2.0], dtype=np.float64))


def test_apply_matrix_to_vector_raises_when_w_is_zero():
    matrix = np.eye(4, dtype=np.float64)
    matrix[3, 3] = 0.0

    with pytest.raises(ValueError):
        apply_matrix_to_vector(matrix, np.array([1.0, 2.0, 3.0], dtype=np.float64))


@pytest.mark.parametrize(
    "angle, expected",
    [
        (-np.pi / 2,
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        ),
        (0.0,
            np.eye(4, dtype=np.float64),
        ),
        (np.pi / 2,
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        ),
    ],
)
def test_rotation_matrix_x(angle, expected):
    result = rotation_matrix_x(angle)
    assert result.shape == (4, 4)
    assert result.dtype == np.float64
    assert np.allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize(
    "angle, expected",
    [
        (-np.pi / 2,
            np.array(
                [
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        ),
        (0.0,
            np.eye(4, dtype=np.float64),
        ),
        (np.pi / 2,
            np.array(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        ),
    ],
)
def test_rotation_matrix_y(angle, expected):
    result = rotation_matrix_y(angle)
    assert result.shape == (4, 4)
    assert result.dtype == np.float64
    assert np.allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize(
    "angle, expected",
    [
        (-np.pi / 2,
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        ),
        (0.0,
            np.eye(4, dtype=np.float64),
        ),
        (np.pi / 2,
            np.array(
                [
                    [0.0, -1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        ),
    ],
)
def test_rotation_matrix_z(angle, expected):
    result = rotation_matrix_z(angle)
    assert result.shape == (4, 4)
    assert result.dtype == np.float64
    assert np.allclose(result, expected, atol=1e-12)