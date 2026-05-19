import numpy as np

from resample3.extrude import Extruder, extrude


def test_extrude_identity_defaults_to_input_shape():
    input_volume = np.array(
        [
            [[1.0, 4.0], [2.0, 5.0]],
            [[3.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)

    output_plane, output_depths = extrude(input_volume, matrix)

    assert output_plane.shape == input_volume.shape[:2]
    assert output_depths.shape == input_volume.shape[:2]
    assert output_plane.dtype == input_volume.dtype
    assert output_depths.dtype == np.float64
    assert np.array_equal(output_plane, input_volume[:, :, 0])
    assert np.array_equal(output_depths, np.zeros(input_volume.shape[:2], dtype=np.float64))


def test_extrude_custom_shape_allocates_requested_outputs():
    input_volume = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    matrix = np.eye(4, dtype=np.float64)

    output_plane, output_depths = extrude(input_volume, matrix, shape=(4, 5))

    assert output_plane.shape == (4, 5)
    assert output_depths.shape == (4, 5)
    assert output_plane.dtype == input_volume.dtype
    assert output_depths.dtype == np.float64


def test_extrude_default_max_depth_for_unfilled_output():
    input_volume = np.array([[[1.0]]], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)

    output_plane, output_depths = extrude(input_volume, matrix, shape=(2, 2))

    expected_max_depth = 2 * max(input_volume.shape)
    assert output_depths[0, 0] == 0.0
    assert np.array_equal(output_depths[1:, :], np.full((1, 2), expected_max_depth))
    assert output_depths[0, 1] == expected_max_depth


def test_extruder_max_depth_override():
    input_volume = np.array([[[3.0]]], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    extruder = Extruder(input_volume, shape=(2, 2), max_depth=7.5)

    _, output_depths = extruder.extrude(matrix)

    assert output_depths[0, 0] == 0.0
    assert output_depths[0, 1] == 7.5
    assert output_depths[1, 0] == 7.5
    assert output_depths[1, 1] == 7.5


def test_extruder_reuses_allocated_buffers():
    input_volume = np.array(
        [
            [[1.0, 9.0], [2.0, 10.0]],
            [[3.0, 11.0], [4.0, 12.0]],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    extruder = Extruder(input_volume)

    output_plane, output_depths = extruder.extrude(matrix)

    assert output_plane is extruder.output_plane
    assert output_depths is extruder.output_depths
    assert np.array_equal(output_plane, input_volume[:, :, 0])
    assert np.array_equal(output_depths, np.zeros(input_volume.shape[:2], dtype=np.float64))
