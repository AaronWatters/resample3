import numpy as np
import pytest

from resample3 import extrude3C


DTYPES = [np.uint8, np.int16, np.int32, np.float32, np.float64]


def extrude_reference(output_shape, input_volume, matrix, min_value):
    min_value_typed = np.array(min_value).astype(input_volume.dtype, casting="unsafe").item()
    output_plane = np.full(output_shape, min_value_typed, dtype=input_volume.dtype)
    output_depths = np.full(output_shape, np.inf, dtype=np.float64)
    src0, src1, src2 = input_volume.shape
    dst0, dst1 = output_shape

    output_i_span = int(np.ceil(np.abs(matrix[0, 0]) + np.abs(matrix[0, 1]) + np.abs(matrix[0, 2])))
    output_j_span = int(np.ceil(np.abs(matrix[1, 0]) + np.abs(matrix[1, 1]) + np.abs(matrix[1, 2])))
    output_i_span = max(output_i_span, 1)
    output_j_span = max(output_j_span, 1)

    for x in range(src0):
        for y in range(src1):
            for z in range(src2):
                val = input_volume[x, y, z]
                if val <= min_value_typed:
                    continue
                coords = matrix @ np.array([x, y, z, 1.0], dtype=np.float64)
                projected_i, projected_j, projected_depth = coords[0], coords[1], coords[2]
                if projected_i < 0.0 or projected_i >= dst0 or projected_j < 0.0 or projected_j >= dst1:
                    continue
                pixel_i_start = int(projected_i)
                pixel_j_start = int(projected_j)
                pixel_i_end = min(pixel_i_start + output_i_span, dst0)
                pixel_j_end = min(pixel_j_start + output_j_span, dst1)
                for pi in range(max(pixel_i_start, 0), pixel_i_end):
                    for pj in range(max(pixel_j_start, 0), pixel_j_end):
                        if projected_depth < output_depths[pi, pj]:
                            output_depths[pi, pj] = projected_depth
                            output_plane[pi, pj] = val
    return output_plane, output_depths


def _min_value(dtype):
    if dtype == np.uint8:
        return 0.0
    return -1.0


@pytest.mark.parametrize("dtype", DTYPES)
def test_extrude_identity(dtype):
    input_volume = (np.arange(27, dtype=np.float64).reshape(3, 3, 3) + 1.0).astype(dtype)
    output_plane = np.empty((3, 3), dtype=dtype)
    output_depths = np.empty((3, 3), dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    min_value = _min_value(dtype)

    extrude3C(output_plane, output_depths, input_volume, matrix, min_value)
    expected_plane, expected_depths = extrude_reference((3, 3), input_volume, matrix, min_value)

    assert np.array_equal(output_plane, expected_plane)
    assert np.array_equal(output_depths, expected_depths)


@pytest.mark.parametrize("dtype", DTYPES)
def test_extrude_rotation_90_degrees(dtype):
    input_volume = (np.arange(27, dtype=np.float64).reshape(3, 3, 3) + 1.0).astype(dtype)
    output_plane = np.empty((3, 3), dtype=dtype)
    output_depths = np.empty((3, 3), dtype=np.float64)
    matrix = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    min_value = _min_value(dtype)

    extrude3C(output_plane, output_depths, input_volume, matrix, min_value)
    expected_plane, expected_depths = extrude_reference((3, 3), input_volume, matrix, min_value)

    assert np.array_equal(output_plane, expected_plane)
    assert np.array_equal(output_depths, expected_depths)


@pytest.mark.parametrize("dtype", DTYPES)
def test_extrude_scaled_identity_no_skips(dtype):
    input_volume = (np.arange(8, dtype=np.float64).reshape(2, 2, 2) + 1.0).astype(dtype)
    output_plane = np.empty((8, 8), dtype=dtype)
    output_depths = np.empty((8, 8), dtype=np.float64)
    matrix = np.array(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    min_value = _min_value(dtype)

    extrude3C(output_plane, output_depths, input_volume, matrix, min_value)
    expected_plane, expected_depths = extrude_reference((8, 8), input_volume, matrix, min_value)

    assert np.array_equal(output_plane, expected_plane)
    assert np.array_equal(output_depths, expected_depths)
    assert np.all(output_plane > np.array(min_value, dtype=dtype))
    assert np.all(np.isfinite(output_depths))
