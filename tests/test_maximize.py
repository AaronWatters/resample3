import numpy as np

from resample3.maximize import Maximizer, maximize


def test_maximize_identity_defaults_to_input_shape():
    input_volume = np.array(
        [
            [[1.0, 4.0], [2.0, 5.0]],
            [[3.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)

    output_plane = maximize(input_volume, matrix)

    assert output_plane.shape == input_volume.shape[:2]
    assert output_plane.dtype == input_volume.dtype
    assert np.array_equal(output_plane, np.max(input_volume, axis=2))


def test_maximize_custom_shape_allocates_requested_output():
    input_volume = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    matrix = np.eye(4, dtype=np.float64)
    min_value = -3.5

    output_plane = maximize(input_volume, matrix, shape=(4, 5), min_value=min_value)

    assert output_plane.shape == (4, 5)
    assert output_plane.dtype == input_volume.dtype
    assert output_plane[3, 4] == np.float32(min_value)


def test_maximizer_reuses_allocated_buffer():
    input_volume = np.array(
        [
            [[1.0, 9.0], [2.0, 10.0]],
            [[3.0, 11.0], [4.0, 12.0]],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    min_value = -7.0
    maximizer = Maximizer(input_volume, shape=(3, 3), min_value=min_value)

    output_plane = maximizer.maximize(matrix)

    assert output_plane is maximizer.output_plane
    assert np.array_equal(output_plane[:2, :2], np.max(input_volume, axis=2))
    assert output_plane[2, 2] == min_value
