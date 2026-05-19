import numpy as np

from resample3.slice import Slicer, slice, slice3py, slicepy


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


def test_slicer_updates_output_with_slice_function():
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    invmatrix = np.eye(4, dtype=np.float64)
    slicer = Slicer(input_volume)
    initial_output_matrix = slicer.output_matrix

    output_matrix = slicer.slice(invmatrix, 1.0)

    assert output_matrix is slicer.output_matrix
    assert output_matrix is not initial_output_matrix
    assert np.array_equal(output_matrix, input_volume[:, :, 1])


def test_slice_allocates_and_returns_output_matrix_using_c_extension():
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    invmatrix = np.eye(4, dtype=np.float64)

    output_matrix = slice(input_volume, 1.0, invmatrix)

    assert output_matrix.shape == input_volume.shape[:2]
    assert np.array_equal(output_matrix, input_volume[:, :, 1])


def test_slice_supports_uint16_dtype_like_project3dc():
    input_volume = np.arange(27, dtype=np.uint16).reshape(3, 3, 3)
    invmatrix = np.eye(4, dtype=np.float64)

    output_matrix = slice(input_volume, 1.0, invmatrix, min_value=0)

    assert output_matrix.dtype == np.uint16
    assert np.array_equal(output_matrix, input_volume[:, :, 1])
