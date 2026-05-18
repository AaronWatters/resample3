#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdint.h>

/*
 * MAX_PROJ_KERNEL(TYPE, SUFFIX)
 *
 * Generates a typed max-value projection function max_proj_SUFFIX.
 *
 * 'm' is the input_to_output_matrix (16 doubles, row-major): it maps input
 * voxel coordinates (x, y, z, 1) to output-plane pixel coordinates.
 */
#define MAX_PROJ_KERNEL(TYPE, SUFFIX)                                               \
static void max_proj_##SUFFIX(                                                      \
    const TYPE *src, TYPE *dst,                                                     \
    npy_intp src0, npy_intp src1, npy_intp src2,                                    \
    npy_intp dst0, npy_intp dst1,                                                   \
    const double *m, TYPE min_val)                                                  \
{                                                                                    \
    for (npy_intp _i = 0; _i < dst0; _i++)                                          \
        for (npy_intp _j = 0; _j < dst1; _j++)                                      \
            dst[_i * dst1 + _j] = min_val;                                          \
    for (npy_intp x = 0; x < src0; x++) {                                           \
        for (npy_intp y = 0; y < src1; y++) {                                       \
            for (npy_intp z = 0; z < src2; z++) {                                   \
                const double pif = m[0] * (double)x + m[1] * (double)y            \
                                 + m[2] * (double)z + m[3];                         \
                const double pjf = m[4] * (double)x + m[5] * (double)y            \
                                 + m[6] * (double)z + m[7];                         \
                const TYPE val = src[(x * src1 + y) * src2 + z];                   \
                if (pif < 0.0 || pif >= (double)dst0 || pjf < 0.0 || pjf >= (double)dst1) \
                    continue;                                                        \
                npy_intp pi0 = (npy_intp)pif;                                        \
                npy_intp pj0 = (npy_intp)pjf;                                        \
                const npy_intp pi_span = (npy_intp)ceil(fabs(m[0]) + fabs(m[1]) + fabs(m[2])); \
                const npy_intp pj_span = (npy_intp)ceil(fabs(m[4]) + fabs(m[5]) + fabs(m[6])); \
                npy_intp pi1 = pi0 + (pi_span > 0 ? pi_span : 1);                    \
                npy_intp pj1 = pj0 + (pj_span > 0 ? pj_span : 1);                    \
                if (pi0 < 0) pi0 = 0;                                               \
                if (pj0 < 0) pj0 = 0;                                               \
                if (pi1 > dst0) pi1 = dst0;                                         \
                if (pj1 > dst1) pj1 = dst1;                                         \
                for (npy_intp pi = pi0; pi < pi1; pi++) {                           \
                    for (npy_intp pj = pj0; pj < pj1; pj++) {                       \
                        TYPE *out_pixel = &dst[pi * dst1 + pj];                     \
                        if (val > *out_pixel)                                        \
                            *out_pixel = val;                                        \
                    }                                                                \
                }                                                                    \
            }                                                                       \
        }                                                                           \
    }                                                                               \
}

MAX_PROJ_KERNEL(double,   double)
MAX_PROJ_KERNEL(float,    float)
MAX_PROJ_KERNEL(uint8_t,  uint8)
MAX_PROJ_KERNEL(uint16_t, uint16)
MAX_PROJ_KERNEL(int16_t,  int16)
MAX_PROJ_KERNEL(int32_t,  int32)

/*
 * EXTRUDE_KERNEL(TYPE, SUFFIX)
 *
 * Generates a typed extrusion function extrude_SUFFIX.
 *
 * 'm' is the input_to_output_matrix (16 doubles, row-major): it maps input
 * voxel coordinates (x, y, z, 1) to output pixel coordinates (pi, pj) and
 * projected depth (pk).
 */
#define EXTRUDE_KERNEL(TYPE, SUFFIX)                                                  \
static void extrude_##SUFFIX(                                                         \
    TYPE *output_plane, double *output_depths,                                        \
    const TYPE *input_volume,                                                         \
    npy_intp output_dim0, npy_intp output_dim1,                                       \
    npy_intp input_dim0, npy_intp input_dim1, npy_intp input_dim2,                    \
    const double *m, TYPE min_val)                                                    \
{                                                                                      \
    for (npy_intp output_i = 0; output_i < output_dim0; output_i++) {                 \
        for (npy_intp output_j = 0; output_j < output_dim1; output_j++) {             \
            const npy_intp output_index = output_i * output_dim1 + output_j;          \
            output_plane[output_index] = min_val;                                      \
            output_depths[output_index] = INFINITY;                                    \
        }                                                                              \
    }                                                                                  \
    const npy_intp output_i_span =                                                     \
        (npy_intp)ceil(fabs(m[0]) + fabs(m[1]) + fabs(m[2]));                         \
    const npy_intp output_j_span =                                                     \
        (npy_intp)ceil(fabs(m[4]) + fabs(m[5]) + fabs(m[6]));                         \
    for (npy_intp input_x = 0; input_x < input_dim0; input_x++) {                     \
        for (npy_intp input_y = 0; input_y < input_dim1; input_y++) {                 \
            for (npy_intp input_z = 0; input_z < input_dim2; input_z++) {             \
                const npy_intp input_index =                                           \
                    (input_x * input_dim1 + input_y) * input_dim2 + input_z;          \
                const TYPE input_value = input_volume[input_index];                    \
                if (input_value <= min_val)                                            \
                    continue;                                                          \
                const double projected_i =                                             \
                    m[0] * (double)input_x + m[1] * (double)input_y                   \
                    + m[2] * (double)input_z + m[3];                                   \
                const double projected_j =                                             \
                    m[4] * (double)input_x + m[5] * (double)input_y                   \
                    + m[6] * (double)input_z + m[7];                                   \
                const double projected_depth =                                         \
                    m[8] * (double)input_x + m[9] * (double)input_y                   \
                    + m[10] * (double)input_z + m[11];                                 \
                if (projected_i < 0.0 || projected_i >= (double)output_dim0           \
                    || projected_j < 0.0 || projected_j >= (double)output_dim1)       \
                    continue;                                                          \
                npy_intp pixel_i_start = (npy_intp)projected_i;                        \
                npy_intp pixel_j_start = (npy_intp)projected_j;                        \
                npy_intp pixel_i_end = pixel_i_start +                                 \
                    (output_i_span > 0 ? output_i_span : 1);                           \
                npy_intp pixel_j_end = pixel_j_start +                                 \
                    (output_j_span > 0 ? output_j_span : 1);                           \
                if (pixel_i_start < 0) pixel_i_start = 0;                              \
                if (pixel_j_start < 0) pixel_j_start = 0;                              \
                if (pixel_i_end > output_dim0) pixel_i_end = output_dim0;              \
                if (pixel_j_end > output_dim1) pixel_j_end = output_dim1;              \
                for (npy_intp output_i = pixel_i_start; output_i < pixel_i_end; output_i++) { \
                    for (npy_intp output_j = pixel_j_start; output_j < pixel_j_end; output_j++) { \
                        const npy_intp output_index = output_i * output_dim1 + output_j; \
                        if (projected_depth < output_depths[output_index]) {            \
                            output_depths[output_index] = projected_depth;              \
                            output_plane[output_index] = input_value;                   \
                        }                                                               \
                    }                                                                   \
                }                                                                       \
            }                                                                          \
        }                                                                              \
    }                                                                                  \
}

EXTRUDE_KERNEL(double,   double)
EXTRUDE_KERNEL(float,    float)
EXTRUDE_KERNEL(uint8_t,  uint8)
EXTRUDE_KERNEL(uint16_t, uint16)
EXTRUDE_KERNEL(int16_t,  int16)
EXTRUDE_KERNEL(int32_t,  int32)

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
        && dtype != NPY_UINT8 && dtype != NPY_UINT16
        && dtype != NPY_INT16 && dtype != NPY_INT32) {
        PyErr_SetString(PyExc_TypeError,
            "unsupported dtype: supported dtypes are uint8, uint16, int16, int32, float32, float64");
        return NULL;
    }

    const double *m = (const double *)PyArray_DATA(matrix);

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
                src0, src1, src2, dst0, dst1, m, (double)min_value);
            break;
        case NPY_FLOAT32:
            max_proj_float(
                (const float *)PyArray_DATA(input),
                (float *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, m, (float)min_value);
            break;
        case NPY_UINT8:
            max_proj_uint8(
                (const uint8_t *)PyArray_DATA(input),
                (uint8_t *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, m, (uint8_t)min_value);
            break;
        case NPY_UINT16:
            max_proj_uint16(
                (const uint16_t *)PyArray_DATA(input),
                (uint16_t *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, m, (uint16_t)min_value);
            break;
        case NPY_INT16:
            max_proj_int16(
                (const int16_t *)PyArray_DATA(input),
                (int16_t *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, m, (int16_t)min_value);
            break;
        case NPY_INT32:
            max_proj_int32(
                (const int32_t *)PyArray_DATA(input),
                (int32_t *)PyArray_DATA(output),
                src0, src1, src2, dst0, dst1, m, (int32_t)min_value);
            break;
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

static PyObject *extrude3C(PyObject *self, PyObject *args)
{
    PyObject *output_plane_obj;
    PyObject *output_depths_obj;
    PyObject *input_volume_obj;
    PyObject *matrix_obj;
    double min_value;

    if (!PyArg_ParseTuple(
            args, "OOOOd",
            &output_plane_obj, &output_depths_obj, &input_volume_obj, &matrix_obj,
            &min_value))
        return NULL;

    if (!PyArray_Check(output_plane_obj) || !PyArray_Check(output_depths_obj)
        || !PyArray_Check(input_volume_obj) || !PyArray_Check(matrix_obj)) {
        PyErr_SetString(
            PyExc_TypeError,
            "output_plane, output_depths, input_volume, and matrix must be numpy arrays");
        return NULL;
    }

    PyArrayObject *output_plane = (PyArrayObject *)output_plane_obj;
    PyArrayObject *output_depths = (PyArrayObject *)output_depths_obj;
    PyArrayObject *input_volume = (PyArrayObject *)input_volume_obj;
    PyArrayObject *matrix = (PyArrayObject *)matrix_obj;

    if (PyArray_NDIM(output_plane) != 2) {
        PyErr_SetString(PyExc_ValueError, "output_plane must be a 2D array");
        return NULL;
    }
    if (PyArray_NDIM(output_depths) != 2) {
        PyErr_SetString(PyExc_ValueError, "output_depths must be a 2D array");
        return NULL;
    }
    if (PyArray_NDIM(input_volume) != 3) {
        PyErr_SetString(PyExc_ValueError, "input_volume must be a 3D array");
        return NULL;
    }
    if (PyArray_NDIM(matrix) != 2
        || PyArray_DIM(matrix, 0) != 4
        || PyArray_DIM(matrix, 1) != 4) {
        PyErr_SetString(PyExc_ValueError, "matrix must be a 4x4 array");
        return NULL;
    }

    if (PyArray_DIM(output_plane, 0) != PyArray_DIM(output_depths, 0)
        || PyArray_DIM(output_plane, 1) != PyArray_DIM(output_depths, 1)) {
        PyErr_SetString(PyExc_ValueError, "output_plane and output_depths must have the same shape");
        return NULL;
    }

    if (!PyArray_CHKFLAGS(output_plane, NPY_ARRAY_C_CONTIGUOUS)
        || !PyArray_CHKFLAGS(output_depths, NPY_ARRAY_C_CONTIGUOUS)
        || !PyArray_CHKFLAGS(input_volume, NPY_ARRAY_C_CONTIGUOUS)
        || !PyArray_CHKFLAGS(matrix, NPY_ARRAY_C_CONTIGUOUS)) {
        PyErr_SetString(
            PyExc_ValueError,
            "output_plane, output_depths, input_volume, and matrix must be C-contiguous");
        return NULL;
    }

    if (PyArray_TYPE(output_depths) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "output_depths dtype must be float64");
        return NULL;
    }
    if (PyArray_TYPE(matrix) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "matrix dtype must be float64");
        return NULL;
    }

    const int dtype = PyArray_TYPE(input_volume);
    if (PyArray_TYPE(output_plane) != dtype) {
        PyErr_SetString(
            PyExc_TypeError,
            "input_volume and output_plane must have the same dtype");
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

    const npy_intp output_dim0 = PyArray_DIM(output_plane, 0);
    const npy_intp output_dim1 = PyArray_DIM(output_plane, 1);
    const npy_intp input_dim0 = PyArray_DIM(input_volume, 0);
    const npy_intp input_dim1 = PyArray_DIM(input_volume, 1);
    const npy_intp input_dim2 = PyArray_DIM(input_volume, 2);
    const double *matrix_data = (const double *)PyArray_DATA(matrix);
    double *output_depths_data = (double *)PyArray_DATA(output_depths);

    Py_BEGIN_ALLOW_THREADS
    switch (dtype) {
        case NPY_FLOAT64:
            extrude_double(
                (double *)PyArray_DATA(output_plane),
                output_depths_data,
                (const double *)PyArray_DATA(input_volume),
                output_dim0, output_dim1,
                input_dim0, input_dim1, input_dim2,
                matrix_data, (double)min_value);
            break;
        case NPY_FLOAT32:
            extrude_float(
                (float *)PyArray_DATA(output_plane),
                output_depths_data,
                (const float *)PyArray_DATA(input_volume),
                output_dim0, output_dim1,
                input_dim0, input_dim1, input_dim2,
                matrix_data, (float)min_value);
            break;
        case NPY_UINT8:
            extrude_uint8(
                (uint8_t *)PyArray_DATA(output_plane),
                output_depths_data,
                (const uint8_t *)PyArray_DATA(input_volume),
                output_dim0, output_dim1,
                input_dim0, input_dim1, input_dim2,
                matrix_data, (uint8_t)min_value);
            break;
        case NPY_UINT16:
            extrude_uint16(
                (uint16_t *)PyArray_DATA(output_plane),
                output_depths_data,
                (const uint16_t *)PyArray_DATA(input_volume),
                output_dim0, output_dim1,
                input_dim0, input_dim1, input_dim2,
                matrix_data, (uint16_t)min_value);
            break;
        case NPY_INT16:
            extrude_int16(
                (int16_t *)PyArray_DATA(output_plane),
                output_depths_data,
                (const int16_t *)PyArray_DATA(input_volume),
                output_dim0, output_dim1,
                input_dim0, input_dim1, input_dim2,
                matrix_data, (int16_t)min_value);
            break;
        case NPY_INT32:
            extrude_int32(
                (int32_t *)PyArray_DATA(output_plane),
                output_depths_data,
                (const int32_t *)PyArray_DATA(input_volume),
                output_dim0, output_dim1,
                input_dim0, input_dim1, input_dim2,
                matrix_data, (int32_t)min_value);
            break;
    }
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

static PyMethodDef MaxProjMethods[] = {
    {"max_value3C", max_value3C, METH_VARARGS,
     "Max-value projection of a 3D volume onto a 2D plane."},
    {"extrude3C", extrude3C, METH_VARARGS,
     "Depth-based extrusion of a 3D volume onto a 2D plane."},
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
