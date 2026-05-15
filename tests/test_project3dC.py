import numpy as np
import pytest

from resample3 import max_value3C


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


def _min_value(dtype):
    """Return a safe min_value sentinel for each dtype."""
    if dtype == np.uint8:
        return 0.0
    return -1.0

# matrix expansion test
def test_matrix_expansion():
    """Test that if the output is bigger thant the input the values infill correctly."""
    input_volume = (np.arange(8, dtype=np.float64).reshape(2, 2, 2)) + 1.0
    output_plane = np.zeros((8, 8), dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64) * 4.0  # scale by 4, so each input voxel maps to a 4x4 block in output
    matrix[3, 3] = 1.0  # homogeneous coordinate
    min_value = 0.0
    max_value3C(input_volume, output_plane, matrix, min_value)
    # output should have no zeros, and each 4x4 block should have the same value corresponding to the input voxel
    assert np.all(output_plane > 0.0)
    for x in range(2):
        for y in range(2):
            val = np.max(input_volume[x, y, :])
            pi_start, pj_start = x * 4, y * 4
            block = output_plane[pi_start:pi_start+4, pj_start:pj_start+4]
            assert np.all(block == val), f"Block at ({pi_start}:{pi_start+4}, {pj_start}:{pj_start+4}) should be {val}"


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
