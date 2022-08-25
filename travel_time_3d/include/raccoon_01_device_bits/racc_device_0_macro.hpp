#pragma once
#ifndef RACC_DEVICE_MACRO
#define RACC_DEVICE_MACRO
#include "_cuda.h"
#include "_racc_device_header_in.h"
namespace racc
{
// CUDA_Stream
#ifndef racc_default_stream
#define racc_default_stream (cudaStream_t)0
#endif
//
#ifndef RACC_BDIM_X
#define RACC_BDIM_X 8
#endif
//
#ifndef RACC_BDIM_Y
#define RACC_BDIM_Y 8
#endif
//
#ifndef RACC_BDIM_Z
#define RACC_BDIM_Z 8
#endif
//
#define RACC_CUDA_IDX_X threadIdx.x + blockDim.x *blockIdx.x;
#define RACC_CUDA_IDX_Y threadIdx.y + blockDim.y *blockIdx.y;
#define RACC_CUDA_IDX_Z threadIdx.z + blockDim.z *blockIdx.z;
//
#ifndef GPU_NUM_PER_NODE
#define GPU_NUM_PER_NODE 4
#endif
//
#define SET_CUDA_FIELD_MESH_IDX(mesh, idx, i, j)     \
    SET_MODEL_MESH_ND(mesh);                         \
    int idx = threadIdx.x + blockDim.x * blockIdx.x; \
    int i = idx % n_rows;                            \
    int j = idx / n_rows
//
#define SET_CUDA_FIELD_GRID_IDX(grid, idx, i, j, k)  \
    SET_MODEL_GRID_ND(grid);                         \
    int idx = threadIdx.x + blockDim.x * blockIdx.x; \
    int i = (idx % (n_elem_slice)) % n_rows;         \
    int j = (idx % (n_elem_slice)) / n_rows;         \
    int k = idx / (n_elem_slice)
//
#define SET_CUDA_TENSOR_GRID_IDX(grid, idx, i, j, k) \
    SET_MODEL_GRID_ND(grid);                         \
    int idx = threadIdx.x + blockDim.x * blockIdx.x; \
    int i = (idx / (mnyz));                          \
    int j = (idx % (mnyz)) / n_slices;               \
    int k = (idx % (mnyz)) % n_slices
//
#define SET_CUDA_MAT_IDX(n_rows, n_cols, idx, i, j)  \
    int idx = threadIdx.x + blockDim.x * blockIdx.x; \
    int i = (idx) % (n_rows);                        \
    int j = (idx) / (n_rows)
//
#define SET_CUDA_CUBE_IDX(n_rows, n_cols, n_slices, idx, i, j, k) \
    int idx = threadIdx.x + blockDim.x * blockIdx.x;              \
    int idx_n_elem_slice = (n_rows) * (n_cols);                   \
    int idx_n_elem = idx_n_elem_slice * (n_slices);               \
    int i = idx % (idx_n_elem_slice) % (n_rows);                  \
    int j = idx % (idx_n_elem_slice) / (n_rows);                  \
    int k = idx / (idx_n_elem_slice)
//
//******************************MARCO_FUN*FOR*GPU**************************************
// The 1D array in CUDA is transformed into the form of 3D ijk display to reduce the code length and enhance the readability, which is used with macro "SET_RAY_MODEL_GRID".
#define F_(a, i, j, k) (a)[(i) + (j)*n_rows + (k)*n_elem_slice] // Field idx
// Finding the smaller value of two numbers.
#define MIN2(x, y) ((x) < (y) ? (x) : (y))
// Finding the minimum of three numbers.
#define MIN3(x, y, z) \
    ((x) < (y) ? (x) : (y)) < (z) ? ((x) < (y) ? (x) : (y)) : (z)
// Exchange the values of the two
#define SWAP(type, a, b) \
    {                    \
        type temp;       \
        temp = a;        \
        a = b;           \
        b = temp;        \
    }

#define CUDA_TIC(a)                       \
    cudaEvent_t startTime##a, endTime##a; \
    cudaEventCreate(&startTime##a);       \
    cudaEventCreate(&endTime##a);         \
    cudaEventRecord(startTime##a, 0)

#define CUDA_TOC(a, str)                                      \
    cudaEventRecord(endTime##a, 0);                           \
    cudaEventSynchronize(startTime##a);                       \
    cudaEventSynchronize(endTime##a);                         \
    float time##a;                                            \
    cudaEventElapsedTime(&time##a, startTime##a, endTime##a); \
    printf(" %s : %f ms \n", str, time##a)

    static void HandleError(cudaError_t err, const char *file, int line)
    {
        if (err != cudaSuccess)
        {
            printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
            exit(EXIT_FAILURE);
        }
    }
//
#define RACC_HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
    //***********************************************************************************
}
#endif