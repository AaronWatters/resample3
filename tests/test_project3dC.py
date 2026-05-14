import numpy as np
import pytest

from resample3 import max_value3C, extrude3C


DTYPES = [np.uint8, np.int16, np.int32, np.float32, np.float64]


def python_reference(input_volume, output_shape, matrix, min_value):
    """Pure-Python reference implementation of max_value3C.

    'matrix' must be the input_to_output_matrix (4x4 float64): applied directly
    to each input voxel coordinate (x, y, z, 1) to obtain the output pixel
    (pi, pj).
    """
    min_val_cast = np.array(min_value).astype(input_volume.dtype, casting="unsafe").item()
    output = np.full(output_shape, min_val_cast, dtype=input_volume.dtype)
    src0, src1, src2 = input_volume.shape
    dst0, dst1 = output_shape
    for x in range(src0):
        for y in range(src1):
            for z in range(src2):
                coords = matrix @ np.array([x, y, z, 1.0], dtype=np.float64)
                pi_f, pj_f = coords[0], coords[1]
                if 0.0 <= pi_f < dst0 and 0.0 <= pj_f < dst1:
                    pi, pj = int(pi_f), int(pj_f)
                    val = input_volume[x, y, z]
                    if val > output[pi, pj]:
                        output[pi, pj] = val
    return output


def python_reference_extrude(input_volume, output_shape, matrix, min_value):
    """Reference for extrude3C using strict depth update (<), preserving first hit on ties."""
    min_val_cast = _cast_scalar_to_dtype(min_value, input_volume.dtype)
    output = np.full(output_shape, min_val_cast, dtype=input_volume.dtype)
    depths = np.full(output_shape, np.inf, dtype=np.float64)
    src0, src1, src2 = input_volume.shape
    dst0, dst1 = output_shape
    for x in range(src0):
        for y in range(src1):
            for z in range(src2):
                val = input_volume[x, y, z]
                if val <= min_val_cast:
                    continue
                coords = matrix @ np.array([x, y, z, 1.0], dtype=np.float64)
                pi_f, pj_f, depth_f = coords[0], coords[1], coords[2]
                if 0.0 <= pi_f < dst0 and 0.0 <= pj_f < dst1:
                    pi, pj = int(pi_f), int(pj_f)
                    if depth_f < depths[pi, pj]:
                        output[pi, pj] = val
                        depths[pi, pj] = depth_f
    return output, depths


def _min_value(dtype):
    """Return a safe min_value sentinel for each dtype."""
    if dtype == np.uint8:
        return 0.0
    return -1.0


def _cast_scalar_to_dtype(value, dtype):
    return np.array(value).astype(dtype, casting="unsafe").item()


# ---------------------------------------------------------------------------
# Identity-mapping tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
def test_identity_mapping(dtype):
    """With an identity matrix, output[i, j] == max over k of input[i, j, k]."""
    input_volume = np.arange(27, dtype=dtype).reshape(3, 3, 3)
    output_plane = np.empty((3, 3), dtype=dtype)
    matrix = np.eye(4, dtype=np.float64)
    min_value = _min_value(dtype)

    max_value3C(input_volume, output_plane, matrix, min_value)

    expected = python_reference(input_volume, (3, 3), matrix, min_value)
    assert np.array_equal(output_plane, expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_extrude_identity_mapping(dtype):
    input_volume = np.arange(8, dtype=dtype).reshape(2, 2, 2)
    output_plane = np.empty((3, 3), dtype=dtype)
    output_depths = np.empty((3, 3), dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    min_value = _min_value(dtype)

    extrude3C(output_plane, output_depths, input_volume, matrix, min_value)

    expected_plane, expected_depths = python_reference_extrude(input_volume, (3, 3), matrix, min_value)
    assert np.array_equal(output_plane, expected_plane)
    assert np.array_equal(output_depths, expected_depths)
    assert output_plane[2, 2] == _cast_scalar_to_dtype(min_value, dtype)
    assert np.isinf(output_depths[2, 2])


# ---------------------------------------------------------------------------
# 90-degree rotation tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
def test_rotation_90_degrees(dtype):
    """A 90-degree rotation around the z-axis produces the correct MIP."""
    input_volume = np.arange(27, dtype=dtype).reshape(3, 3, 3)
    output_plane = np.empty((3, 3), dtype=dtype)
    # input_to_output_matrix: maps input (x, y, z) -> output (pi=y, pj=2-x)
    matrix = np.array(
        [
            [ 0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 2.0],
            [ 0.0, 0.0, 1.0, 0.0],
            [ 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    min_value = _min_value(dtype)

    max_value3C(input_volume, output_plane, matrix, min_value)

    expected = python_reference(input_volume, (3, 3), matrix, min_value)
    assert np.array_equal(output_plane, expected)


@pytest.mark.parametrize("dtype", DTYPES)
def test_extrude_rotation_90_degrees(dtype):
    input_volume = np.arange(8, dtype=dtype).reshape(2, 2, 2)
    output_plane = np.empty((4, 4), dtype=dtype)
    output_depths = np.empty((4, 4), dtype=np.float64)
    matrix = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    min_value = _min_value(dtype)

    extrude3C(output_plane, output_depths, input_volume, matrix, min_value)

    expected_plane, expected_depths = python_reference_extrude(input_volume, (4, 4), matrix, min_value)
    assert np.array_equal(output_plane, expected_plane)
    assert np.array_equal(output_depths, expected_depths)
    assert output_plane[3, 3] == _cast_scalar_to_dtype(min_value, dtype)
    assert np.isinf(output_depths[3, 3])


@pytest.mark.parametrize("dtype", DTYPES)
def test_extrude_depth_buffer_prefers_smaller_depth(dtype):
    input_volume = np.array([[[5, 9, 7]]], dtype=dtype)
    output_plane = np.empty((1, 1), dtype=dtype)
    output_depths = np.empty((1, 1), dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    min_value = _min_value(dtype)

    extrude3C(output_plane, output_depths, input_volume, matrix, min_value)

    # All voxels project to (0, 0); the smallest depth is z=0, so value=5 should win.
    assert output_plane[0, 0] == _cast_scalar_to_dtype(5, dtype)
    assert output_depths[0, 0] == 0.0


# ---------------------------------------------------------------------------
# Error / validation tests
# ---------------------------------------------------------------------------

def test_raises_for_wrong_input_ndim():
    input_volume = np.arange(9, dtype=np.float64).reshape(3, 3)
    output_plane = np.empty((3, 3), dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    with pytest.raises(ValueError):
        max_value3C(input_volume, output_plane, matrix, 0.0)


def test_raises_for_wrong_output_ndim():
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    output_plane = np.empty((3, 3, 3), dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    with pytest.raises(ValueError):
        max_value3C(input_volume, output_plane, matrix, 0.0)


def test_raises_for_dtype_mismatch():
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    output_plane = np.empty((3, 3), dtype=np.float32)
    matrix = np.eye(4, dtype=np.float64)
    with pytest.raises(TypeError):
        max_value3C(input_volume, output_plane, matrix, 0.0)


def test_raises_for_non_contiguous_input():
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)[:, :, ::-1]
    output_plane = np.empty((3, 3), dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    with pytest.raises(ValueError):
        max_value3C(input_volume, output_plane, matrix, 0.0)


def test_raises_for_non_contiguous_output():
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    output_plane = np.empty((3, 6), dtype=np.float64)[:, ::2]
    matrix = np.eye(4, dtype=np.float64)
    with pytest.raises(ValueError):
        max_value3C(input_volume, output_plane, matrix, 0.0)


def test_extrude_raises_for_wrong_input_ndim():
    output_plane = np.empty((3, 3), dtype=np.float64)
    output_depths = np.empty((3, 3), dtype=np.float64)
    input_volume = np.arange(9, dtype=np.float64).reshape(3, 3)
    matrix = np.eye(4, dtype=np.float64)
    with pytest.raises(ValueError):
        extrude3C(output_plane, output_depths, input_volume, matrix, 0.0)


def test_extrude_raises_for_wrong_output_ndim():
    output_plane = np.empty((3, 3, 1), dtype=np.float64)
    output_depths = np.empty((3, 3), dtype=np.float64)
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    matrix = np.eye(4, dtype=np.float64)
    with pytest.raises(ValueError):
        extrude3C(output_plane, output_depths, input_volume, matrix, 0.0)


def test_extrude_raises_for_depth_shape_mismatch():
    output_plane = np.empty((3, 3), dtype=np.float64)
    output_depths = np.empty((2, 3), dtype=np.float64)
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    matrix = np.eye(4, dtype=np.float64)
    with pytest.raises(ValueError):
        extrude3C(output_plane, output_depths, input_volume, matrix, 0.0)


def test_extrude_raises_for_dtype_mismatch():
    output_plane = np.empty((3, 3), dtype=np.float32)
    output_depths = np.empty((3, 3), dtype=np.float64)
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    matrix = np.eye(4, dtype=np.float64)
    with pytest.raises(TypeError):
        extrude3C(output_plane, output_depths, input_volume, matrix, 0.0)


def test_extrude_raises_for_non_float64_depths():
    output_plane = np.empty((3, 3), dtype=np.float64)
    output_depths = np.empty((3, 3), dtype=np.float32)
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    matrix = np.eye(4, dtype=np.float64)
    with pytest.raises(TypeError):
        extrude3C(output_plane, output_depths, input_volume, matrix, 0.0)


def test_extrude_raises_for_non_contiguous_input():
    output_plane = np.empty((3, 3), dtype=np.float64)
    output_depths = np.empty((3, 3), dtype=np.float64)
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)[:, :, ::-1]
    matrix = np.eye(4, dtype=np.float64)
    with pytest.raises(ValueError):
        extrude3C(output_plane, output_depths, input_volume, matrix, 0.0)


def test_extrude_raises_for_non_contiguous_output():
    output_plane = np.empty((3, 6), dtype=np.float64)[:, ::2]
    output_depths = np.empty((3, 3), dtype=np.float64)
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    matrix = np.eye(4, dtype=np.float64)
    with pytest.raises(ValueError):
        extrude3C(output_plane, output_depths, input_volume, matrix, 0.0)


def test_extrude_raises_for_non_contiguous_depths():
    output_plane = np.empty((3, 3), dtype=np.float64)
    output_depths = np.empty((3, 6), dtype=np.float64)[:, ::2]
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    matrix = np.eye(4, dtype=np.float64)
    with pytest.raises(ValueError):
        extrude3C(output_plane, output_depths, input_volume, matrix, 0.0)


def test_extrude_raises_for_wrong_matrix_shape():
    output_plane = np.empty((3, 3), dtype=np.float64)
    output_depths = np.empty((3, 3), dtype=np.float64)
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    matrix = np.eye(3, dtype=np.float64)
    with pytest.raises(ValueError):
        extrude3C(output_plane, output_depths, input_volume, matrix, 0.0)


def test_extrude_raises_for_wrong_matrix_dtype():
    output_plane = np.empty((3, 3), dtype=np.float64)
    output_depths = np.empty((3, 3), dtype=np.float64)
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    matrix = np.eye(4, dtype=np.float32)
    with pytest.raises(TypeError):
        extrude3C(output_plane, output_depths, input_volume, matrix, 0.0)
