#pragma once
#ifndef RACC_TMEMORY_MEAT
#define RACC_TMEMORY_MEAT
#include "racc_host_30_Tmemory_bones.hpp"
//*********************************************************************************
//************************Memory allocation and release of array*******************
// Allocate one dimensional array memory.
namespace racc
{
    template <typename T>
    T *alloc1d(uint64_t l)
    {
        if (l > 0)
        {
            T *a = nullptr;
            a = (T *)calloc(l, sizeof(T));
            return a;
        }
    }
    template <typename T>
    T *new1d(uint64_t l)
    {
        if (l > 0)
        {
            T *a = nullptr;
            a = new T[l];
            return a;
        }
    }
    template <typename T>
    void alloc1d(T *&a, uint64_t l)
    {
        if (l > 0)
        {
            a = (T *)calloc(l, sizeof(T));
        }
    }
    template <typename T>
    void new1d(T *&a, uint64_t l)
    {
        if (l > 0)
        {
            a = new T[l];
        }
    }
    // Allocate memory for 2D arrays.
    template <typename T>
    T **alloc2d(uint32_t n_rows, uint32_t n_cols)
    {
        T **a = nullptr;
        uint32_t i;
        a = (T **)calloc(n_rows, sizeof(T *));
        if (a)
            for (i = 0; i < n_rows; i++)
            {
                a[i] = (T *)calloc(n_cols, sizeof(T));
            }
        return a;
    }
    template <typename T>
    T **new2d(uint32_t n_rows, uint32_t n_cols)
    {
        T **a = nullptr;
        uint32_t i;
        a = new T *[n_rows];
        for (i = 0; i < n_rows; i++)
        {
            a[i] = new T[n_cols];
        }
        return a;
    }

    template <typename T>
    void alloc2d(T **&a, uint32_t n_rows, uint32_t n_cols)
    {
        a = (T **)calloc(n_rows, sizeof(T *));
        for (uint32_t i = 0; i < n_rows; i++)
        {
            a[i] = (T *)calloc(n_cols, sizeof(T));
        }
    }
    template <typename T>
    void new2d(T **&a, uint32_t n_rows, uint32_t n_cols)
    {
        a = new T *[n_rows];
        for (uint32_t i = 0; i < n_rows; i++)
        {
            a[i] = new T[n_cols];
        }
    }
    // Allocate 3D array memory.
    template <typename T>
    T ***alloc3d(uint32_t n_rows, uint32_t n_cols, uint32_t n_slices)
    {
        T ***a = nullptr;
        a = (T ***)calloc(n_rows, sizeof(T **));
        if (a)
            for (uint32_t i = 0; i < n_rows; i++)
            {
                a[i] = (T **)calloc(n_cols, sizeof(T *));
                if (a[i])
                    for (uint32_t j = 0; j < n_cols; j++)
                        a[i][j] = (T *)calloc(n_slices, sizeof(T));
            }
        return a;
    }
    template <typename T>
    T ***new3d(uint32_t n_rows, uint32_t n_cols, uint32_t n_slices)
    {
        T ***a = nullptr;
        a = new T **[n_rows];
        for (uint32_t i = 0; i < n_rows; i++)
        {
            a[i] = new T *[n_cols];
            for (uint32_t j = 0; j < n_cols; j++)
            {
                a[i][j] = new T[n_slices];
            }
        }
        return a;
    }
    template <typename T>
    void alloc3d(T ***&a, uint32_t n_rows, uint32_t n_cols, uint32_t n_slices)
    {
        a = (T ***)calloc(n_rows, sizeof(T **));
        for (uint32_t i = 0; i < n_rows; i++)
        {
            a[i] = (T **)calloc(n_cols, sizeof(T *));
            for (uint32_t j = 0; j < n_cols; j++)
                a[i][j] = (T *)calloc(n_slices, sizeof(T));
        }
    }
    template <typename T>
    void new3d(T ***&a, uint32_t n_rows, uint32_t n_cols, uint32_t n_slices)
    {
        a = new T **[n_rows];
        for (uint32_t i = 0; i < n_rows; i++)
        {
            a[i] = new T *[n_cols];
            for (uint32_t j = 0; j < n_cols; j++)
            {
                a[i][j] = new T[n_slices];
            }
        }
    }

    // Free one dimensional array memory.
    template <typename T>
    void free1d(T *&a)
    {
        free(a);
        a = 0;
    }
    template <typename T>
    void del1d(T *&a, uint64_t l)
    {
        delete[] a;
        a = 0;
    }
    // Free 2D array memory.
    template <typename T>
    void free2d(T **a, uint32_t _row_l, uint32_t _row_r)
    {
        for (uint32_t i = _row_l; i <= _row_r; i++)
        {
            free(a[i]);
            a[i] = 0;
        }
        free(a);
        a = 0;
    }
    template <typename T>
    void del2d(T **&a, uint32_t n_rows)
    {
        for (uint32_t i = 0; i < n_rows; i++)
        {
            delete[] a[i];
            a[i] = 0;
        }
        delete[] a;
        a = 0;
    }

    // Free 3D array memory.
    template <typename T>
    void free3d(T ***&a, uint32_t n_rows, uint32_t n_cols)
    {
        for (uint32_t i = 0; i < n_rows; i++)
            for (uint32_t j = 0; j < n_cols; j++)
            {
                free(a[i][j]);
                a[i][j] = 0;
            }
        for (uint32_t i = 0; i < n_rows; i++)
        {
            free(a[i]);
            a[i] = 0;
        }
        free(a);
        a = 0;
    }
    template <typename T>
    void del3d(T ***&a, uint32_t n_rows, uint32_t n_cols, uint32_t n_slices)
    {
        for (uint32_t i = 0; i < n_rows; i++)
            for (uint32_t j = 0; j < n_cols; j++)
            {
                delete[] a[i][j];
                a[i][j] = 0;
            }
        for (uint32_t i = 0; i < n_rows; i++)
        {
            delete[] a[i];
            a[i] = 0;
        }
        delete[] a;
        a = 0;
    }
    //
    template <typename T>
    T *ralloc1d(int n_l, int n_r)
    {
        T *a = nullptr;
        a = (T *)calloc(n_r - n_l + 1 + NR_END, sizeof(T));
        return a - n_l + NR_END;
    }
    template <typename T>
    T **ralloc2d(int row_l, int row_r, int col_l, int col_r)
    {
        int nrow = row_r - row_l + 1, ncol = col_r - col_l + 1;
        T **a = nullptr;
        /* allocate pointers to rows */
        a = (T **)calloc(nrow + NR_END, sizeof(T *));
        a += NR_END;
        a -= row_l;
        /* allocation rows and set pointers to them */
        if (a)
            a[row_l] = (T *)calloc(nrow * ncol + NR_END, sizeof(T));
        a[row_l] += NR_END;
        a[row_l] -= col_l;
        for (int i = row_l + 1; i <= row_r; i++)
            a[i] = a[i - 1] + ncol;
        /* return pointer to array of pointer to rows */
        return a;
    }
    template <typename T>
    T ***ralloc3d(int row_l, int row_r, int col_l, int col_r, int slice_l, int slice_r)
    {
        int n_row = row_r - row_l + 1, n_col = col_r - col_l + 1, n_slice = slice_r - slice_l + 1;
        T ***a = nullptr;
        /* allocate pointers to pointers to rows */
        a = (T ***)calloc(n_row + NR_END, sizeof(T **));
        a += NR_END;
        a -= row_l;
        /* allocate pointers to rows and set pointers to them */
        if (a)
            a[row_l] = (T **)calloc(n_row * n_col + NR_END, sizeof(T *));
        a[row_l] += NR_END;
        a[row_l] -= col_l;

        /* allocate rows and set pointers to them */
        a[row_l][col_l] = (T *)calloc(n_row * n_col * n_slice + NR_END, sizeof(T));
        a[row_l][col_l] += NR_END;
        a[row_l][col_l] -= slice_l;

        for (int j = col_l + 1; j <= col_r; j++)
            a[row_l][j] = a[row_l][j - 1] + n_slice;
        for (int i = row_l + 1; i <= row_r; i++)
        {
            a[i] = a[i - 1] + n_col;
            a[i][col_l] = a[i - 1][col_l] + n_col * n_slice;
            for (int j = col_l + 1; j <= col_r; j++)
                a[i][j] = a[i][j - 1] + n_slice;
        }
        /* return pointer to array of pointer to rows */
        return a;
    }
    template <typename T>
    void rfree1d(T *v, int n_l, int n_r)
    {
        /* free a float vector allocated with vector() */
        free((FREE_ARG)(v + n_l - NR_END));
    }
    template <typename T>
    void rfree2d(T **m, int row_l, int row_r, int col_l, int col_r)
    {
        /* free a float matrix allocated by matrix() */
        free((FREE_ARG)(m[row_l] + col_l - NR_END));
        free((FREE_ARG)(m + row_l - NR_END));
    }
    template <typename T>
    void rfree3d(T ***t, int row_l, int row_r, int col_l, int col_r, int slice_l, int slice_r)
    {
        /* free a T matrix allocated by f3tensor() */
        free((FREE_ARG)(t[row_l][col_l] + slice_l - NR_END));
        free((FREE_ARG)(t[row_l] + col_l - NR_END));
        free((FREE_ARG)(t + row_l - NR_END));
    }
}
#endif