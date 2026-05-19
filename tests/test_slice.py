import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _load_slicepy():
    repo_root = Path(__file__).resolve().parents[1]
    package_name = "slicepy_test_module"
    package_path = repo_root / "resample3"

    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules[package_name] = package

    matrices_name = f"{package_name}.matrices"
    matrices_spec = importlib.util.spec_from_file_location(
        matrices_name, package_path / "matrices.py"
    )
    matrices_module = importlib.util.module_from_spec(matrices_spec)
    sys.modules[matrices_name] = matrices_module
    matrices_spec.loader.exec_module(matrices_module)

    slice_name = f"{package_name}.slice"
    slice_spec = importlib.util.spec_from_file_location(slice_name, package_path / "slice.py")
    slice_module = importlib.util.module_from_spec(slice_spec)
    sys.modules[slice_name] = slice_module
    slice_spec.loader.exec_module(slice_module)
    return slice_module.slicepy


@pytest.fixture
def slicepy():
    return _load_slicepy()


def test_slicepy_identity_defaults_to_input_xy_shape(slicepy):
    input_volume = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    invmatrix = np.eye(4, dtype=np.float64)

    output_plane = slicepy(input_volume, depth=1, invmatrix=invmatrix)

    assert output_plane.shape == input_volume.shape[:2]
    assert output_plane.dtype == input_volume.dtype
    assert np.array_equal(output_plane, input_volume[:, :, 1])


def test_slicepy_custom_shape_uses_min_value_for_clamp_and_out_of_bounds(slicepy):
    input_volume = np.array(
        [
            [[-2.0, 5.0], [1.0, 4.0]],
            [[0.25, 3.0], [2.0, 6.0]],
        ],
        dtype=np.float32,
    )
    invmatrix = np.eye(4, dtype=np.float64)
    min_value = 0.5

    output_plane = slicepy(
        input_volume,
        depth=0,
        invmatrix=invmatrix,
        shape=(3, 3),
        min_value=min_value,
    )

    expected = np.full((3, 3), np.float32(min_value), dtype=np.float32)
    expected[:2, :2] = np.maximum(np.float32(min_value), input_volume[:, :, 0])

    assert output_plane.shape == (3, 3)
    assert output_plane.dtype == input_volume.dtype
    assert np.array_equal(output_plane, expected)
