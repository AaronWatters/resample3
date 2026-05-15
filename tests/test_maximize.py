import numpy as np
import pytest

from resample3 import maximize, Maximizer


DTYPES = [np.uint8, np.int16, np.int32, np.float32, np.float64]


def _min_value(dtype):
    if dtype == np.uint8:
        return 0.0
    return -1.0


def reference_maximize(input_volume, matrix, shape, min_value):
    """Pure-Python reference matching max_proj span-fill behaviour in the C kernel."""
    min_val_cast = np.array(min_value).astype(input_volume.dtype, casting="unsafe").item()
    output = np.full(shape, min_val_cast, dtype=input_volume.dtype)
    src0, src1, src2 = input_volume.shape
    dst0, dst1 = shape
    pi_span = int(np.ceil(abs(matrix[0, 0]) + abs(matrix[0, 1]) + abs(matrix[0, 2])))
    pj_span = int(np.ceil(abs(matrix[1, 0]) + abs(matrix[1, 1]) + abs(matrix[1, 2])))
    pi_span = max(pi_span, 1)
    pj_span = max(pj_span, 1)
    for x in range(src0):
        for y in range(src1):
            for z in range(src2):
                coords = matrix @ np.array([x, y, z, 1.0], dtype=np.float64)
                pi_f, pj_f = coords[0], coords[1]
                if pi_f < 0.0 or pi_f >= dst0 or pj_f < 0.0 or pj_f >= dst1:
                    continue
                pi0 = int(pi_f)
                pj0 = int(pj_f)
                val = input_volume[x, y, z]
                for pi in range(max(pi0, 0), min(pi0 + pi_span, dst0)):
                    for pj in range(max(pj0, 0), min(pj0 + pj_span, dst1)):
                        if val > output[pi, pj]:
                            output[pi, pj] = val
    return output


@pytest.mark.parametrize("dtype", DTYPES)
def test_maximize_identity(dtype):
    input_volume = (np.arange(27, dtype=np.float64).reshape(3, 3, 3) + 1.0).astype(dtype)
    matrix = np.eye(4, dtype=np.float64)
    min_value = _min_value(dtype)

    result = maximize(input_volume, matrix, min_value=min_value)

    expected = reference_maximize(input_volume, matrix, (3, 3), min_value)
    assert np.array_equal(result, expected)
    assert result.shape == (3, 3)
    assert result.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_maximize_explicit_shape(dtype):
    input_volume = (np.arange(27, dtype=np.float64).reshape(3, 3, 3) + 1.0).astype(dtype)
    matrix = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    shape = (6, 6)
    min_value = _min_value(dtype)

    result = maximize(input_volume, matrix, shape=shape, min_value=min_value)

    expected = reference_maximize(input_volume, matrix, shape, min_value)
    assert np.array_equal(result, expected)
    assert result.shape == shape
    assert result.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_maximizer_class(dtype):
    input_volume = (np.arange(27, dtype=np.float64).reshape(3, 3, 3) + 1.0).astype(dtype)
    matrix = np.eye(4, dtype=np.float64)
    min_value = _min_value(dtype)

    m = Maximizer(input_volume, matrix, min_value=min_value)
    result = m.maximize()

    expected = reference_maximize(input_volume, matrix, (3, 3), min_value)
    assert np.array_equal(result, expected)
    assert result.shape == (3, 3)
    assert result.dtype == dtype


def test_maximizer_reuses_output_buffer():
    input_volume = (np.arange(27, dtype=np.float64).reshape(3, 3, 3) + 1.0)
    matrix = np.eye(4, dtype=np.float64)

    m = Maximizer(input_volume, matrix)
    result1 = m.maximize()
    result2 = m.maximize()
    assert result1 is result2


def test_maximize_default_shape():
    input_volume = np.ones((4, 5, 6), dtype=np.float32)
    matrix = np.eye(4, dtype=np.float64)

    result = maximize(input_volume, matrix)
    assert result.shape == (4, 5)
