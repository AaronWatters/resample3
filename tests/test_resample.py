import numpy as np

from resample3.resample import sample, Sampler


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


def test_sampler_defaults_to_input_shape():
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    s = Sampler(input_volume)
    assert s.output_volume.shape == input_volume.shape
    assert s.output_volume.dtype == input_volume.dtype


def test_sampler_custom_shape():
    input_volume = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    s = Sampler(input_volume, shape=(2, 2, 2))
    assert s.output_volume.shape == (2, 2, 2)
    assert s.output_volume.dtype == np.float32


def test_sampler_sample_identity():
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    s = Sampler(input_volume)
    matrix = np.eye(4, dtype=np.float64)
    result = s.sample(matrix)
    assert result.shape == input_volume.shape
    assert np.array_equal(result, input_volume)


def test_sampler_sample_non_contiguous_inputs():
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)[:, :, ::-1]
    matrix = np.eye(4, dtype=np.float64).T
    s = Sampler(input_volume)
    result = s.sample(matrix)
    assert np.array_equal(result, np.ascontiguousarray(input_volume))


def test_sampler_returns_output_volume():
    input_volume = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    s = Sampler(input_volume)
    matrix = np.eye(4, dtype=np.float64)
    result = s.sample(matrix)
    assert result is s.output_volume

