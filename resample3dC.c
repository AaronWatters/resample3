#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>

#define RESAMPLE_KERNEL(TYPE, SUFFIX)                                                   \
static void resample_##SUFFIX(                                                          \
    TYPE *src, TYPE *dst,                                                               \
    npy_intp src0, npy_intp src1, npy_intp src2,                                        \
    npy_intp dst0, npy_intp dst1, npy_intp dst2,                                        \
    const double *m, TYPE default_value)                                                \
{                                                                                        \
    for (npy_intp i = 0; i < dst0; ++i) {                                               \
        for (npy_intp j = 0; j < dst1; ++j) {                                           \
            for (npy_intp k = 0; k < dst2; ++k) {                                       \
                const npy_intp out_index = (i * dst1 + j) * dst2 + k;                   \
                const double x = m[0] * (double)i + m[1] * (double)j + m[2] * (double)k + m[3]; \
                const double y = m[4] * (double)i + m[5] * (double)j + m[6] * (double)k + m[7]; \
                const double z = m[8] * (double)i + m[9] * (double)j + m[10] * (double)k + m[11]; \
                if (x <= (double)NPY_MIN_INTP || x >= (double)NPY_MAX_INTP              \
                    || y <= (double)NPY_MIN_INTP || y >= (double)NPY_MAX_INTP           \
                    || z <= (double)NPY_MIN_INTP || z >= (double)NPY_MAX_INTP) {        \
                    dst[out_index] = default_value;                                      \
                    continue;                                                            \
                }                                                                        \
                const npy_intp ii = (npy_intp)x;                                         \
                const npy_intp jj = (npy_intp)y;                                         \
                const npy_intp kk = (npy_intp)z;                                         \
                if (ii >= 0 && jj >= 0 && kk >= 0                                        \
                    && ii < src0 && jj < src1 && kk < src2) {                           \
                    const npy_intp in_index = (ii * src1 + jj) * src2 + kk;             \
                    dst[out_index] = src[in_index];                                      \
                } else {                                                                 \
                    dst[out_index] = default_value;                                      \
                }                                                                        \
            }                                                                            \
        }                                                                                \
    }                                                                                    \
}

RESAMPLE_KERNEL(double, double)
RESAMPLE_KERNEL(float, float)
RESAMPLE_KERNEL(uint8_t, uint8)
RESAMPLE_KERNEL(int16_t, int16)
RESAMPLE_KERNEL(int32_t, int32)

static PyObject *resample3C(PyObject *self, PyObject *args)
{
    PyObject *input_obj;
    PyObject *output_obj;
    PyObject *matrix_obj;
    double default_value;

    if (!PyArg_ParseTuple(args, "OOOd", &input_obj, &output_obj, &matrix_obj, &default_value)) {
        return NULL;
    }

    if (!PyArray_Check(input_obj) || !PyArray_Check(output_obj) || !PyArray_Check(matrix_obj)) {
        PyErr_SetString(PyExc_TypeError, "input_volume, output_volume, and matrix must be numpy arrays");
        return NULL;
    }

    PyArrayObject *input = (PyArrayObject *)input_obj;
    PyArrayObject *output = (PyArrayObject *)output_obj;
    PyArrayObject *matrix = (PyArrayObject *)matrix_obj;

    if (PyArray_NDIM(input) != 3 || PyArray_NDIM(output) != 3) {
        PyErr_SetString(PyExc_ValueError, "input_volume and output_volume must be 3D arrays");
        return NULL;
    }
    if (PyArray_NDIM(matrix) != 2 || PyArray_DIM(matrix, 0) != 4 || PyArray_DIM(matrix, 1) != 4) {
        PyErr_SetString(PyExc_ValueError, "matrix must be a 4x4 array");
        return NULL;
    }

    if (!PyArray_CHKFLAGS(input, NPY_ARRAY_C_CONTIGUOUS)
        || !PyArray_CHKFLAGS(output, NPY_ARRAY_C_CONTIGUOUS)
        || !PyArray_CHKFLAGS(matrix, NPY_ARRAY_C_CONTIGUOUS)) {
        PyErr_SetString(PyExc_ValueError, "input_volume, output_volume, and matrix must be C-contiguous");
        return NULL;
    }

    if (PyArray_TYPE(matrix) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "matrix dtype must be float64");
        return NULL;
    }

    const int dtype = PyArray_TYPE(input);
    if (PyArray_TYPE(output) != dtype) {
        PyErr_SetString(PyExc_TypeError, "input_volume and output_volume must have the same dtype");
        return NULL;
    }

    const npy_intp src0 = PyArray_DIM(input, 0);
    const npy_intp src1 = PyArray_DIM(input, 1);
    const npy_intp src2 = PyArray_DIM(input, 2);
    const npy_intp dst0 = PyArray_DIM(output, 0);
    const npy_intp dst1 = PyArray_DIM(output, 1);
    const npy_intp dst2 = PyArray_DIM(output, 2);
    const double *m = (const double *)PyArray_DATA(matrix);

    switch (dtype) {
        case NPY_FLOAT64:
            resample_double(
                (double *)PyArray_DATA(input), (double *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, dst2, m, (double)default_value);
            break;
        case NPY_FLOAT32:
            resample_float(
                (float *)PyArray_DATA(input), (float *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, dst2, m, (float)default_value);
            break;
        case NPY_UINT8:
            resample_uint8(
                (uint8_t *)PyArray_DATA(input), (uint8_t *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, dst2, m, (uint8_t)default_value);
            break;
        case NPY_INT16:
            resample_int16(
                (int16_t *)PyArray_DATA(input), (int16_t *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, dst2, m, (int16_t)default_value);
            break;
        case NPY_INT32:
            resample_int32(
                (int32_t *)PyArray_DATA(input), (int32_t *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, dst2, m, (int32_t)default_value);
            break;
        default:
            PyErr_SetString(
                PyExc_TypeError,
                "unsupported dtype: supported dtypes are uint8, int16, int32, float32, float64");
            return NULL;
    }

    Py_RETURN_NONE;
}

static PyMethodDef ResampleMethods[] = {
    {"resample3C", resample3C, METH_VARARGS, "Resample 3D volume with nearest-neighbor interpolation."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef resample3Cmodule = {
    PyModuleDef_HEAD_INIT,
    "resample3C",
    NULL,
    -1,
    ResampleMethods
};

PyMODINIT_FUNC PyInit_resample3C(void)
{
    import_array();
    return PyModule_Create(&resample3Cmodule);
}
