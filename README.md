# resample3
Map volume output coordinates to input coordinates using nearest neighbor interpolation using a matrix.

## Build

```bash
python -m pip install -e .
```

## Test

```bash
python -m pip install pytest numpy
pytest -q
```

## Demos

The demos require the `H5Gizmos` library.

```bash
pip install resample3[demos]
```


