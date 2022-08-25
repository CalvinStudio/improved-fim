#pragma once
#ifndef RACC_TMEMORY_BONES
#define RACC_TMEMORY_BONES
#include "racc_host_21_struct_meat.hpp"
//*********************************************************************************
//************************Memory allocation and release of array*******************
// Allocate one dimensional array memory.
namespace racc
{
#define NR_END (0)
#define FREE_ARG char *
    template <typename T>
    T *alloc1d(uint64_t l);
    template <typename T>
    T *new1d(uint64_t l);
    template <typename T>
    void alloc1d(T *&a, uint64_t l);
    template <typename T>
    void new1d(T *&a, uint64_t l);
    // Allocate memory for 2D arrays.
    template <typename T>
    T **alloc2d(uint32_t n_rows, uint32_t n_cols);
    template <typename T>
    T **new2d(uint32_t n_rows, uint32_t n_cols);
    template <typename T>
    void alloc2d(T **&a, uint32_t n_rows, uint32_t n_cols);
    template <typename T>
    void new2d(T **&a, uint32_t n_rows, uint32_t n_cols);
    // Allocate 3D array memory.
    template <typename T>
    T ***alloc3d(uint32_t n_rows, uint32_t n_cols, uint32_t n_slices);
    template <typename T>
    T ***new3d(uint32_t n_rows, uint32_t n_cols, uint32_t n_slices);
    template <typename T>
    void alloc3d(T ***&a, uint32_t n_rows, uint32_t n_cols, uint32_t n_slices);
    template <typename T>
    void new3d(T ***&a, uint32_t n_rows, uint32_t n_cols, uint32_t n_slices);
    // Free one dimensional array memory.
    template <typename T>
    void free1d(T *&a);
    template <typename T>
    void del1d(T *&a, uint32_t l);
    // Free 2D array memory.
    template <typename T>
    void free2d(T **&a, uint32_t n_rows);
    template <typename T>
    void del2d(T **&a, uint32_t n_rows, uint32_t n_cols);
    // Free 3D array memory.
    template <typename T>
    void free3d(T ***&a, uint32_t n_rows, uint32_t n_cols);
    template <typename T>
    void del3d(T ***&a, uint32_t n_rows, uint32_t n_cols, uint32_t n_slices);
    //
    template <typename T = float>
    T *ralloc1d(int n_l, int n_r);
    template <typename T = float>
    T **ralloc2d(int row_l, int row_r, int col_l, int col_r);
    template <typename T = float>
    T ***ralloc3d(int row_l, int row_r, int col_l, int col_r, int slice_l, int slice_r);
    //
    template <typename T>
    void rfree1d(T *v, int n_l, int n_r);
    template <typename T>
    void rfree2d(T **m, int row_l, int row_r, int col_l, int col_r);
    template <typename T = float>
    void rfree3d(T ***t, int row_l, int row_r, int col_l, int col_r, int slice_l, int slice_r);
}
#endif