import numpy as np

from resample3.matrices import translation_matrix
from resample3.slice import slicepy


def test_slicepy_identity_uses_depth_and_default_shape():
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    invmatrix = np.eye(4, dtype=np.float64)

    output = slicepy(input_volume, depth=1, invmatrix=invmatrix)

    assert output.shape == input_volume.shape[:2]
    assert output.dtype == input_volume.dtype
    assert np.array_equal(output, input_volume[:, :, 1])


def test_slicepy_custom_shape_fills_out_of_bounds_with_min_value():
    input_volume = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    invmatrix = np.eye(4, dtype=np.float64)

    output = slicepy(input_volume, depth=0, invmatrix=invmatrix, shape=(3, 4), min_value=-1.0)

    expected = np.full((3, 4), -1.0, dtype=np.float64)
    expected[:2, :2] = input_volume[:, :, 0]
    assert np.array_equal(output, expected)


def test_slicepy_clamps_values_to_min_value():
    input_volume = np.array(
        [
            [[-3.0], [2.0]],
            [[-1.0], [4.0]],
        ],
        dtype=np.float64,
    )
    invmatrix = np.eye(4, dtype=np.float64)

    output = slicepy(input_volume, depth=0, invmatrix=invmatrix, min_value=0.5)

    expected = np.array([[0.5, 2.0], [0.5, 4.0]], dtype=np.float64)
    assert np.array_equal(output, expected)


def test_slicepy_translation_out_of_bounds_returns_min_value_everywhere():
    input_volume = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    invmatrix = translation_matrix(10.0, 10.0, 0.0)

    output = slicepy(input_volume, depth=0, invmatrix=invmatrix, min_value=-2.0)

    assert np.array_equal(output, np.full(input_volume.shape[:2], -2.0, dtype=np.float64))
