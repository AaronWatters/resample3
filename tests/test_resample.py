import numpy as np

from resample3.resample import sample


def test_sample_defaults_to_input_shape():
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    matrix = np.eye(4, dtype=np.float64)
    output_volume = sample(input_volume, matrix)
    assert output_volume.shape == input_volume.shape
    assert np.array_equal(output_volume, input_volume)


def test_sample_accepts_non_contiguous_inputs():
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)[:, :, ::-1]
    matrix = np.eye(4, dtype=np.float64).T
    output_volume = sample(input_volume, matrix)
    assert np.array_equal(output_volume, input_volume)

