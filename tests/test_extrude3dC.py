import numpy as np
import pytest

from resample3 import extrude3C


DTYPES = [np.uint8, np.int16, np.int32, np.float32, np.float64]


def python_extrude_reference(output_shape, input_volume, matrix, min_value):
    min_val_cast = np.array(min_value).astype(input_volume.dtype, casting="unsafe").item()
    output_plane = np.full(output_shape, min_val_cast, dtype=input_volume.dtype)
    output_depths = np.full(output_shape, np.inf, dtype=np.float64)
    src0, src1, src2 = input_volume.shape
    dst0, dst1 = output_shape

    pi_span = int(np.ceil(np.abs(matrix[0, 0]) + np.abs(matrix[0, 1]) + np.abs(matrix[0, 2])))
    pj_span = int(np.ceil(np.abs(matrix[1, 0]) + np.abs(matrix[1, 1]) + np.abs(matrix[1, 2])))
    pi_span = max(pi_span, 1)
    pj_span = max(pj_span, 1)

    for x in range(src0):
        for y in range(src1):
            for z in range(src2):
                val = input_volume[x, y, z]
                if val <= min_val_cast:
                    continue
                coords = matrix @ np.array([x, y, z, 1.0], dtype=np.float64)
                pi_f, pj_f, pk_f = coords[0], coords[1], coords[2]
                if pi_f < 0.0 or pi_f >= dst0 or pj_f < 0.0 or pj_f >= dst1:
                    continue
                pi0 = int(pi_f)
                pj0 = int(pj_f)
                pi1 = min(pi0 + pi_span, dst0)
                pj1 = min(pj0 + pj_span, dst1)
                for pi in range(max(pi0, 0), pi1):
                    for pj in range(max(pj0, 0), pj1):
                        if pk_f < output_depths[pi, pj]:
                            output_depths[pi, pj] = pk_f
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
    expected_plane, expected_depths = python_extrude_reference((3, 3), input_volume, matrix, min_value)

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
    expected_plane, expected_depths = python_extrude_reference((3, 3), input_volume, matrix, min_value)

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
    expected_plane, expected_depths = python_extrude_reference((8, 8), input_volume, matrix, min_value)

    assert np.array_equal(output_plane, expected_plane)
    assert np.array_equal(output_depths, expected_depths)
    assert np.all(output_plane > np.array(min_value, dtype=dtype))
    assert np.all(np.isfinite(output_depths))
