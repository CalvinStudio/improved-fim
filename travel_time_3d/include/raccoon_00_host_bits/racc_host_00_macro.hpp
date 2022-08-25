#pragma once
#ifndef RACC_MACRO_H
#define RACC_MACRO_H
//**********************************Developer******************************************
// 2020.04.10 BY CAIWEI CALVIN CAI
//*************************************************************************************
#include "_racc_host_header_in.h"
//********************************MACRO_DEFINE*****************************************
namespace racc
{
  typedef complex<float> cx_float;
  typedef complex<double> cx_double;
  typedef unsigned int uint;
//*************************************************************************************

//********************************MACRO_FUNCTION***************************************
//*******************************MARCO_FUN*FOR*CPU*************************************
// Assignment of temporary variables in mesh or grid model.
#define SET_MODEL_MESH(Mesh)                            \
  int n_rows = (Mesh).n_rows, n_cols = (Mesh).n_cols;   \
  float l_rows = (Mesh).l_rows, l_cols = (Mesh).l_cols; \
  float d_rows = (Mesh).d_rows, d_cols = (Mesh).d_cols;

#define SET_MODEL_MESH_ND(Mesh)                       \
  int n_rows = (Mesh).n_rows, n_cols = (Mesh).n_cols; \
  float d_rows = (Mesh).d_rows, d_cols = (Mesh).d_cols;

#define SET_MODEL_GRID_N(Grid)                                                    \
  int n_rows = (Grid).n_rows, n_cols = (Grid).n_cols, n_slices = (Grid).n_slices; \
  int n_elem_slice = (Grid).n_elem_slice, mnyz = n_cols * n_slices

#define SET_MODEL_GRID_ND(Grid)                                                     \
  int n_rows = (Grid).n_rows, n_cols = (Grid).n_cols, n_slices = (Grid).n_slices;   \
  float d_rows = (Grid).d_rows, d_cols = (Grid).d_cols, d_slices = (Grid).d_slices; \
  int n_elem_slice = (Grid).n_elem_slice

#define SET_PRE_MODEL_GRID_ND(Pre, Grid)                                                            \
  int Pre##_n_rows = (Grid).n_rows, Pre##_n_cols = (Grid).n_cols, Pre##_n_slices = (Grid).n_slices; \
  int Pre##_n_elem_slice = (Grid).n_elem_slice;

#define SET_MODEL_GRID_LR(Grid)                                                     \
  float l_rows = (Grid).l_rows, l_cols = (Grid).l_cols, l_slices = (Grid).l_slices; \
  float r_rows = (Grid).r_rows, r_cols = (Grid).r_cols, r_slices = (Grid).r_slices;

#define SET_MODEL_GRID_NDL(Grid)                                                    \
  int n_rows = (Grid).n_rows, n_cols = (Grid).n_cols, n_slices = (Grid).n_slices;   \
  float d_rows = (Grid).d_rows, d_cols = (Grid).d_cols, d_slices = (Grid).d_slices; \
  float l_rows = (Grid).l_rows, l_cols = (Grid).l_cols, l_slices = (Grid).l_slices; \
  int n_elem_slice = (Grid).n_elem_slice;

#define SET_MODEL_GRID(Grid)                                                        \
  int n_rows = (Grid).n_rows, n_cols = (Grid).n_cols, n_slices = (Grid).n_slices;   \
  float d_rows = (Grid).d_rows, d_cols = (Grid).d_cols, d_slices = (Grid).d_slices; \
  float l_rows = (Grid).l_rows, l_cols = (Grid).l_cols, l_slices = (Grid).l_slices; \
  float r_rows = (Grid).r_rows, r_cols = (Grid).r_cols, r_slices = (Grid).r_slices;

#define _CFOR(I, J, K)          \
  for (int k = 0; k < K; k++)   \
    for (int j = 0; j < J; j++) \
      for (int i = 0; i < I; i++)

#define FOR_SIZEOF(A)                  \
  for (int k = 0; k < A.n_slices; k++) \
    for (int j = 0; j < A.n_cols; j++) \
      for (int i = 0; i < A.n_rows; i++)

//*************************************************************************************
#define MAT_FOR2D                      \
  for (int i = row_l; i <= row_r; i++) \
    for (int j = col_l; j <= col_r; j++)

#define MESH_FOR2D(mesh)                  \
  for (int i = 0; i < (mesh).n_rows; i++) \
    for (int j = 0; j < (mesh).n_cols; j++)

#define MESH_FOR_IDX2D(mesh)              \
  for (int j = 0; j < (mesh).n_cols; j++) \
    for (int i = 0; i < (mesh).n_rows; i++)

#define MESH_IDX2D(mesh) i + j *(mesh).n_rows

//*************************************************************************************
#define CUBE_FOR3D                                           \
  for (int i = idx_range.row_l; i <= idx_range.row_r; i++)   \
    for (int j = idx_range.col_l; j <= idx_range.col_r; j++) \
      for (int k = idx_range.slice_l; k <= idx_range.slice_r; k++)

#define CUBE_FOR_IDX3D               \
  for (int k = 0; k < n_slices; k++) \
    for (int j = 0; j < n_cols; j++) \
      for (int i = 0; i < n_rows; i++)

#define CUBE_IDX3D i + j *n_rows + k *n_rows *n_cols

#define GRID_FOR3D(grid)                    \
  for (int i = 0; i < (grid).n_rows; i++)   \
    for (int j = 0; j < (grid).n_cols; j++) \
      for (int k = 0; k < (grid).n_slices; k++)

#define GRID_FOR_IDX3D(grid)                \
  for (int k = 0; k < (grid).n_slices; k++) \
    for (int j = 0; j < (grid).n_cols; j++) \
      for (int i = 0; i < (grid).n_rows; i++)

#define GRID_IDX3D(grid) i + j *(grid).n_rows + k *(grid).n_rows *(grid).n_cols

#define GMS_IDX3D(grid) i *(grid).n_slices + j *(grid).n_rows *(grid).n_slices + k

//**********************One dimension to multi dimension index********************
#define FIELD_FOR                          \
  for (int k = 0; k < frame.n_slices; k++) \
    for (int j = 0; j < frame.n_cols; j++) \
      for (int i = 0; i < frame.n_rows; i++)

#define FIELD_IDX i + j *frame.n_rows + k *frame.n_elem_slice

#define FRAME_FOR(frame)                     \
  for (int k = 0; k < (frame).n_slices; k++) \
    for (int j = 0; j < (frame).n_cols; j++) \
      for (int i = 0; i < (frame).n_rows; i++)

#define FRAME_IDX(frame) i + j *(frame).n_rows + k *(frame).n_elem_slice

  //*************************************************************************************

#define PRINT_GRID_INFO(grid)                                                                                  \
  cout << "Grid_X_Num:" << setw(5) << grid.n_rows << "; INTERVAL:" << setw(6) << grid.d_rows << "; RANGE:"     \
       << "[" << grid.l_rows << ", " << grid.r_rows << "]" << endl;                                            \
  cout << "Grid_Y_Num:" << setw(5) << grid.n_cols << "; INTERVAL:" << setw(6) << grid.d_cols << "; RANGE:"     \
       << "[" << grid.l_cols << ", " << grid.r_cols << "]" << endl;                                            \
  cout << "Grid_Z_Num:" << setw(5) << grid.n_slices << "; INTERVAL:" << setw(6) << grid.d_slices << "; RANGE:" \
       << "[" << grid.l_slices << ", " << grid.r_slices << "]" << endl;

}
#endif