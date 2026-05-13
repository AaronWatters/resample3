import numpy as np
import pytest

import resample3dC


DTYPES = [np.uint8, np.int16, np.int32, np.float32, np.float64]


def python_reference(input_volume, output_shape, matrix, default_value):
    output = np.empty(output_shape, dtype=input_volume.dtype)
    src0, src1, src2 = input_volume.shape
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            for k in range(output_shape[2]):
                x = matrix[0, 0] * i + matrix[0, 1] * j + matrix[0, 2] * k + matrix[0, 3]
                y = matrix[1, 0] * i + matrix[1, 1] * j + matrix[1, 2] * k + matrix[1, 3]
                z = matrix[2, 0] * i + matrix[2, 1] * j + matrix[2, 2] * k + matrix[2, 3]
                ii, jj, kk = int(x), int(y), int(z)
                if 0 <= ii < src0 and 0 <= jj < src1 and 0 <= kk < src2:
                    output[i, j, k] = input_volume[ii, jj, kk]
                else:
                    output[i, j, k] = np.array(default_value, dtype=input_volume.dtype)
    return output


@pytest.mark.parametrize("dtype", DTYPES)
def test_identity_mapping(dtype):
    input_volume = np.arange(27, dtype=np.int64).reshape(3, 3, 3).astype(dtype)
    output_volume = np.empty((3, 3, 3), dtype=dtype)
    matrix = np.eye(4, dtype=np.float64)
    default_value = -7.25

    resample3dC.resample3C(input_volume, output_volume, matrix, default_value)

    assert np.array_equal(output_volume, input_volume)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rotation_90_degrees_about_z(dtype):
    input_volume = np.arange(27, dtype=np.int64).reshape(3, 3, 3).astype(dtype)
    output_volume = np.empty((3, 3, 3), dtype=dtype)
    matrix = np.array(
        [
            [0.0, -1.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    default_value = -3.5

    resample3dC.resample3C(input_volume, output_volume, matrix, default_value)
    expected = python_reference(input_volume, output_volume.shape, matrix, default_value)

    assert np.array_equal(output_volume, expected)


def test_raises_for_non_contiguous_array():
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)[:, :, ::-1]
    output_volume = np.empty((3, 3, 3), dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)

    with pytest.raises(ValueError):
        resample3dC.resample3C(input_volume, output_volume, matrix, 0.0)


def test_raises_for_dtype_mismatch():
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    output_volume = np.empty((3, 3, 3), dtype=np.float32)
    matrix = np.eye(4, dtype=np.float64)

    with pytest.raises(TypeError):
        resample3dC.resample3C(input_volume, output_volume, matrix, 0.0)
