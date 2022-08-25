#ifndef RACC_HOST_TIO_BONES
#define RACC_HOST_TIO_BONES
#include "racc_host_51_fun_meat.hpp"
namespace racc
{
    // Template
    // Input
    template <typename T>
    void ReadGrdFile(string pathname, MeshFrame &mesh, T **&grddata);
    template <typename T>
    T ***InputTensor(string _path, uint32_t _n_rows, uint32_t _n_cols, uint32_t &_n_slices, int _format, bool _isdim = false);
    // Output
    template <typename T>
    void MatOutPut(string _path, T **_data, uint32_t _n_rows, uint32_t _n_cols, int _row_l = 0, int _col_l = 0, int _format = 0);
    template <typename T>
    void MeshOutPut(string file_path, T **Mat, MeshFrame Mesh);
    template <typename T>
    void CubeOutPut(string file_path, T ***Cube, uint32_t n_rows, uint32_t n_cols, uint32_t n_slices);
    template <typename T>
    void CubeOutPut(string file_path, string FileName, T ***Cube, uint32_t n_rows, uint32_t n_cols, uint32_t n_slices);
    template <typename T>
    void GridSliceOutPut(string file_path, GridFrame Grid, T ***Cube, string Axis, int Num);
    template <typename T>
    void GridSliceOutPut(string file_path, string FileName, GridFrame Grid, T ***Cube, string Axis, int Num);
    template <typename T>
    void GridOutPut(string file_path, T ***Cube, GridFrame grid);
    template <typename T>
    void GridOutPut(string file_path, string FileName, T ***Cube, GridFrame grid);
    template <typename T>
    void OutputTensor(string _path, T ***_data, uint32_t _n_rows, uint32_t _n_cols, uint32_t _n_slices, int _row_l = 0, int _col_l = 0, int _nz1 = 0, int _format = 0);
    template <typename T>
    void OutputVector(string _path, T *_data, uint32_t _n_rows, int _row_l = 0, int _format = 0);
    // Print
    template <typename T>
    void vecprint_number(int64_t n_l, int64_t n_r, T *vec);
    template <typename T>
    void vecprint_struct(int64_t n_l, int64_t n_r, T *vec);
    template <typename T>
    void matprint(T **mat, uint32_t n_rows, uint32_t n_cols);
}
#endif