#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>

#define SLICE_KERNEL(TYPE, SUFFIX)                                                        \
static void slice_##SUFFIX(                                                               \
    const TYPE *src, TYPE *dst,                                                           \
    npy_intp src0, npy_intp src1, npy_intp src2,                                          \
    npy_intp dst0, npy_intp dst1,                                                         \
    const double *m, double depth, TYPE min_val)                                          \
{                                                                                          \
    for (npy_intp i = 0; i < dst0; i++) {                                                 \
        for (npy_intp j = 0; j < dst1; j++) {                                             \
            dst[i * dst1 + j] = min_val;                                                  \
            const double x = m[0] * (double)i + m[1] * (double)j + m[2] * depth + m[3];  \
            const double y = m[4] * (double)i + m[5] * (double)j + m[6] * depth + m[7];  \
            const double z = m[8] * (double)i + m[9] * (double)j + m[10] * depth + m[11];\
            if (x <= (double)NPY_MIN_INTP || x >= (double)NPY_MAX_INTP                    \
                || y <= (double)NPY_MIN_INTP || y >= (double)NPY_MAX_INTP                 \
                || z <= (double)NPY_MIN_INTP || z >= (double)NPY_MAX_INTP)                \
                continue;                                                                  \
            const npy_intp ii = (npy_intp)x;                                              \
            const npy_intp jj = (npy_intp)y;                                              \
            const npy_intp kk = (npy_intp)z;                                              \
            if (ii >= 0 && ii < src0 && jj >= 0 && jj < src1 && kk >= 0 && kk < src2) {  \
                const TYPE val = src[(ii * src1 + jj) * src2 + kk];                       \
                dst[i * dst1 + j] = (val > min_val) ? val : min_val;                      \
            }                                                                              \
        }                                                                                  \
    }                                                                                      \
}

SLICE_KERNEL(double,   double)
SLICE_KERNEL(float,    float)
SLICE_KERNEL(uint8_t,  uint8)
SLICE_KERNEL(uint16_t, uint16)
SLICE_KERNEL(int16_t,  int16)
SLICE_KERNEL(int32_t,  int32)

static PyObject *slice3dC(PyObject *self, PyObject *args)
{
    PyObject *output_obj;
    PyObject *input_obj;
    double depth;
    PyObject *matrix_obj;
    double min_value;

    if (!PyArg_ParseTuple(args, "OOdOd", &output_obj, &input_obj, &depth, &matrix_obj, &min_value))
        return NULL;

    if (!PyArray_Check(output_obj) || !PyArray_Check(input_obj) || !PyArray_Check(matrix_obj)) {
        PyErr_SetString(
            PyExc_TypeError,
            "output_matrix, input_volume, and matrix must be numpy arrays");
        return NULL;
    }

    PyArrayObject *output = (PyArrayObject *)output_obj;
    PyArrayObject *input = (PyArrayObject *)input_obj;
    PyArrayObject *matrix = (PyArrayObject *)matrix_obj;

    if (PyArray_NDIM(output) != 2) {
        PyErr_SetString(PyExc_ValueError, "output_matrix must be a 2D array");
        return NULL;
    }
    if (PyArray_NDIM(input) != 3) {
        PyErr_SetString(PyExc_ValueError, "input_volume must be a 3D array");
        return NULL;
    }
    if (PyArray_NDIM(matrix) != 2
        || PyArray_DIM(matrix, 0) != 4
        || PyArray_DIM(matrix, 1) != 4) {
        PyErr_SetString(PyExc_ValueError, "matrix must be a 4x4 array");
        return NULL;
    }

    if (!PyArray_CHKFLAGS(output, NPY_ARRAY_C_CONTIGUOUS)
        || !PyArray_CHKFLAGS(input, NPY_ARRAY_C_CONTIGUOUS)
        || !PyArray_CHKFLAGS(matrix, NPY_ARRAY_C_CONTIGUOUS)) {
        PyErr_SetString(
            PyExc_ValueError,
            "output_matrix, input_volume, and matrix must be C-contiguous");
        return NULL;
    }

    if (PyArray_TYPE(matrix) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "matrix dtype must be float64");
        return NULL;
    }

    const int dtype = PyArray_TYPE(input);
    if (PyArray_TYPE(output) != dtype) {
        PyErr_SetString(
            PyExc_TypeError,
            "input_volume and output_matrix must have the same dtype");
        return NULL;
    }
    if (dtype != NPY_FLOAT64 && dtype != NPY_FLOAT32
        && dtype != NPY_UINT8 && dtype != NPY_UINT16
        && dtype != NPY_INT16 && dtype != NPY_INT32) {
        PyErr_SetString(
            PyExc_TypeError,
            "unsupported dtype: supported dtypes are uint8, uint16, int16, int32, float32, float64");
        return NULL;
    }

    const npy_intp src0 = PyArray_DIM(input, 0);
    const npy_intp src1 = PyArray_DIM(input, 1);
    const npy_intp src2 = PyArray_DIM(input, 2);
    const npy_intp dst0 = PyArray_DIM(output, 0);
    const npy_intp dst1 = PyArray_DIM(output, 1);
    const double *m = (const double *)PyArray_DATA(matrix);

    Py_BEGIN_ALLOW_THREADS
    switch (dtype) {
        case NPY_FLOAT64:
            slice_double(
                (const double *)PyArray_DATA(input),
                (double *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, m, depth, (double)min_value);
            break;
        case NPY_FLOAT32:
            slice_float(
                (const float *)PyArray_DATA(input),
                (float *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, m, depth, (float)min_value);
            break;
        case NPY_UINT8:
            slice_uint8(
                (const uint8_t *)PyArray_DATA(input),
                (uint8_t *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, m, depth, (uint8_t)min_value);
            break;
        case NPY_UINT16:
            slice_uint16(
                (const uint16_t *)PyArray_DATA(input),
                (uint16_t *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, m, depth, (uint16_t)min_value);
            break;
        case NPY_INT16:
            slice_int16(
                (const int16_t *)PyArray_DATA(input),
                (int16_t *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, m, depth, (int16_t)min_value);
            break;
        case NPY_INT32:
            slice_int32(
                (const int32_t *)PyArray_DATA(input),
                (int32_t *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, m, depth, (int32_t)min_value);
            break;
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

static PyMethodDef SliceMethods[] = {
    {"slice3dC", slice3dC, METH_VARARGS, "Slice a 3D volume onto a 2D matrix."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef slice3dCmodule = {
    PyModuleDef_HEAD_INIT,
    "slice3dC",
    NULL,
    -1,
    SliceMethods
};

PyMODINIT_FUNC PyInit_slice3dC(void)
{
    import_array();
    return PyModule_Create(&slice3dCmodule);
}
