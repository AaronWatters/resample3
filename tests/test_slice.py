import numpy as np

from resample3.slice import slice3py, slicepy


def test_slice3py_writes_into_preallocated_output_matrix():
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    output_matrix = np.empty((3, 3), dtype=np.float64)
    invmatrix = np.eye(4, dtype=np.float64)

    result = slice3py(output_matrix, input_volume, 1.0, invmatrix, min_value=-1.0)

    assert result is None
    assert np.array_equal(output_matrix, input_volume[:, :, 1])


def test_slicepy_allocates_and_returns_output_matrix():
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    invmatrix = np.eye(4, dtype=np.float64)

    output_matrix = slicepy(input_volume, 1.0, invmatrix)

    assert output_matrix.shape == input_volume.shape[:2]
    assert np.array_equal(output_matrix, input_volume[:, :, 1])