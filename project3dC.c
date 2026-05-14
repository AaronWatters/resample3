#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <math.h>

/*
 * Compute the inverse of a 4x4 row-major matrix m into inv_out using the
 * cofactor / adjugate method.  Returns 1 on success, 0 if the matrix is
 * (near-)singular.
 */
static int mat4_inverse(const double *m, double *inv_out)
{
    double inv[16];

    inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15]
             + m[9]*m[7]*m[14]  + m[13]*m[6]*m[11]  - m[13]*m[7]*m[10];
    inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14]  + m[8]*m[6]*m[15]
             - m[8]*m[7]*m[14]  - m[12]*m[6]*m[11]  + m[12]*m[7]*m[10];
    inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13]  - m[8]*m[5]*m[15]
             + m[8]*m[7]*m[13]  + m[12]*m[5]*m[11]  - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13]  + m[8]*m[5]*m[14]
             - m[8]*m[6]*m[13]  - m[12]*m[5]*m[10]  + m[12]*m[6]*m[9];

    inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14]  + m[9]*m[2]*m[15]
             - m[9]*m[3]*m[14]  - m[13]*m[2]*m[11]  + m[13]*m[3]*m[10];
    inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14]  - m[8]*m[2]*m[15]
             + m[8]*m[3]*m[14]  + m[12]*m[2]*m[11]  - m[12]*m[3]*m[10];
    inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13]  + m[8]*m[1]*m[15]
             - m[8]*m[3]*m[13]  - m[12]*m[1]*m[11]  + m[12]*m[3]*m[9];
    inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13]  - m[8]*m[1]*m[14]
             + m[8]*m[2]*m[13]  + m[12]*m[1]*m[10]  - m[12]*m[2]*m[9];

    inv[2]  =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]   - m[5]*m[2]*m[15]
             + m[5]*m[3]*m[14]  + m[13]*m[2]*m[7]   - m[13]*m[3]*m[6];
    inv[6]  = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]   + m[4]*m[2]*m[15]
             - m[4]*m[3]*m[14]  - m[12]*m[2]*m[7]   + m[12]*m[3]*m[6];
    inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]   - m[4]*m[1]*m[15]
             + m[4]*m[3]*m[13]  + m[12]*m[1]*m[7]   - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]   + m[4]*m[1]*m[14]
             - m[4]*m[2]*m[13]  - m[12]*m[1]*m[6]   + m[12]*m[2]*m[5];

    inv[3]  = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]   + m[5]*m[2]*m[11]
             - m[5]*m[3]*m[10]  - m[9]*m[2]*m[7]    + m[9]*m[3]*m[6];
    inv[7]  =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]   - m[4]*m[2]*m[11]
             + m[4]*m[3]*m[10]  + m[8]*m[2]*m[7]    - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]    + m[4]*m[1]*m[11]
             - m[4]*m[3]*m[9]   - m[8]*m[1]*m[7]    + m[8]*m[3]*m[5];
    inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]    - m[4]*m[1]*m[10]
             + m[4]*m[2]*m[9]   + m[8]*m[1]*m[6]    - m[8]*m[2]*m[5];

    const double det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    if (fabs(det) < 1e-12)
        return 0;

    const double inv_det = 1.0 / det;
    for (int i = 0; i < 16; i++)
        inv_out[i] = inv[i] * inv_det;
    return 1;
}

/*
 * MAX_PROJ_KERNEL(TYPE, SUFFIX)
 *
 * Generates a typed max-value projection function max_proj_SUFFIX.
 *
 * The function:
 *   1. Fills the output plane with min_val.
 *   2. Iterates over every voxel in the input volume.
 *   3. Maps each voxel's index through the inverse of output_to_input_matrix
 *      (stored in mi, 16 doubles, row-major) to obtain a 2-D output-plane
 *      pixel index (pi, pj).
 *   4. If (pi, pj) is within the output plane, stores the voxel value when
 *      it is greater than the current value at that pixel.
 */
#define MAX_PROJ_KERNEL(TYPE, SUFFIX)                                               \
static void max_proj_##SUFFIX(                                                      \
    const TYPE *src, TYPE *dst,                                                     \
    npy_intp src0, npy_intp src1, npy_intp src2,                                    \
    npy_intp dst0, npy_intp dst1,                                                   \
    const double *mi, TYPE min_val)                                                 \
{                                                                                    \
    for (npy_intp _i = 0; _i < dst0; _i++)                                          \
        for (npy_intp _j = 0; _j < dst1; _j++)                                      \
            dst[_i * dst1 + _j] = min_val;                                          \
    for (npy_intp x = 0; x < src0; x++) {                                           \
        for (npy_intp y = 0; y < src1; y++) {                                       \
            for (npy_intp z = 0; z < src2; z++) {                                   \
                const double pif = mi[0]*(double)x + mi[1]*(double)y               \
                                 + mi[2]*(double)z + mi[3];                         \
                const double pjf = mi[4]*(double)x + mi[5]*(double)y               \
                                 + mi[6]*(double)z + mi[7];                         \
                if (pif < 0.0 || pif >= (double)dst0                                \
                    || pjf < 0.0 || pjf >= (double)dst1)                            \
                    continue;                                                        \
                const npy_intp pi = (npy_intp)pif;                                  \
                const npy_intp pj = (npy_intp)pjf;                                  \
                const TYPE val = src[(x * src1 + y) * src2 + z];                   \
                TYPE *out_pixel = &dst[pi * dst1 + pj];                            \
                if (val > *out_pixel)                                               \
                    *out_pixel = val;                                               \
            }                                                                       \
        }                                                                           \
    }                                                                               \
}

MAX_PROJ_KERNEL(double,   double)
MAX_PROJ_KERNEL(float,    float)
MAX_PROJ_KERNEL(uint8_t,  uint8)
MAX_PROJ_KERNEL(int16_t,  int16)
MAX_PROJ_KERNEL(int32_t,  int32)

static PyObject *max_value3C(PyObject *self, PyObject *args)
{
    PyObject *input_obj;
    PyObject *output_obj;
    PyObject *matrix_obj;
    double min_value;

    if (!PyArg_ParseTuple(args, "OOOd", &input_obj, &output_obj, &matrix_obj, &min_value))
        return NULL;

    if (!PyArray_Check(input_obj) || !PyArray_Check(output_obj) || !PyArray_Check(matrix_obj)) {
        PyErr_SetString(PyExc_TypeError,
            "input_volume, output_plane, and matrix must be numpy arrays");
        return NULL;
    }

    PyArrayObject *input  = (PyArrayObject *)input_obj;
    PyArrayObject *output = (PyArrayObject *)output_obj;
    PyArrayObject *matrix = (PyArrayObject *)matrix_obj;

    if (PyArray_NDIM(input) != 3) {
        PyErr_SetString(PyExc_ValueError, "input_volume must be a 3D array");
        return NULL;
    }
    if (PyArray_NDIM(output) != 2) {
        PyErr_SetString(PyExc_ValueError, "output_plane must be a 2D array");
        return NULL;
    }
    if (PyArray_NDIM(matrix) != 2
        || PyArray_DIM(matrix, 0) != 4
        || PyArray_DIM(matrix, 1) != 4) {
        PyErr_SetString(PyExc_ValueError, "matrix must be a 4x4 array");
        return NULL;
    }

    if (!PyArray_CHKFLAGS(input, NPY_ARRAY_C_CONTIGUOUS)
        || !PyArray_CHKFLAGS(output, NPY_ARRAY_C_CONTIGUOUS)
        || !PyArray_CHKFLAGS(matrix, NPY_ARRAY_C_CONTIGUOUS)) {
        PyErr_SetString(PyExc_ValueError,
            "input_volume, output_plane, and matrix must be C-contiguous");
        return NULL;
    }

    if (PyArray_TYPE(matrix) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "matrix dtype must be float64");
        return NULL;
    }

    const int dtype = PyArray_TYPE(input);
    if (PyArray_TYPE(output) != dtype) {
        PyErr_SetString(PyExc_TypeError,
            "input_volume and output_plane must have the same dtype");
        return NULL;
    }

    if (dtype != NPY_FLOAT64 && dtype != NPY_FLOAT32
        && dtype != NPY_UINT8 && dtype != NPY_INT16 && dtype != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError,
            "unsupported dtype: supported dtypes are uint8, int16, int32, float32, float64");
        return NULL;
    }

    const double *m = (const double *)PyArray_DATA(matrix);
    double mi[16];
    if (!mat4_inverse(m, mi)) {
        PyErr_SetString(PyExc_ValueError,
            "output_to_input_matrix is singular or near-singular");
        return NULL;
    }

    const npy_intp src0 = PyArray_DIM(input, 0);
    const npy_intp src1 = PyArray_DIM(input, 1);
    const npy_intp src2 = PyArray_DIM(input, 2);
    const npy_intp dst0 = PyArray_DIM(output, 0);
    const npy_intp dst1 = PyArray_DIM(output, 1);

    Py_BEGIN_ALLOW_THREADS
    switch (dtype) {
        case NPY_FLOAT64:
            max_proj_double(
                (const double *)PyArray_DATA(input),
                (double *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, mi, (double)min_value);
            break;
        case NPY_FLOAT32:
            max_proj_float(
                (const float *)PyArray_DATA(input),
                (float *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, mi, (float)min_value);
            break;
        case NPY_UINT8:
            max_proj_uint8(
                (const uint8_t *)PyArray_DATA(input),
                (uint8_t *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, mi, (uint8_t)min_value);
            break;
        case NPY_INT16:
            max_proj_int16(
                (const int16_t *)PyArray_DATA(input),
                (int16_t *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, mi, (int16_t)min_value);
            break;
        case NPY_INT32:
            max_proj_int32(
                (const int32_t *)PyArray_DATA(input),
                (int32_t *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, mi, (int32_t)min_value);
            break;
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

static PyMethodDef MaxProjMethods[] = {
    {"max_value3C", max_value3C, METH_VARARGS,
     "Max-value projection of a 3D volume onto a 2D plane."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef project3dCmodule = {
    PyModuleDef_HEAD_INIT,
    "project3dC",
    NULL,
    -1,
    MaxProjMethods
};

PyMODINIT_FUNC PyInit_project3dC(void)
{
    import_array();
    return PyModule_Create(&project3dCmodule);
}
