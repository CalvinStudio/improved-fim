#pragma once
#ifndef RACC_FUN_BONES
#define RACC_FUN_BONES
//**********************************Developer****************************************
// 2020.04.10 BY CAIWEI CALVIN CAI
//*******************************************************************************
#include "racc_host_41_Tfun_meat.hpp"
//************************************文件安全相关*********************************
// Check if file is opened successfully.
namespace racc
{
	// Check if filename length out of limit.
	inline void FileNameTooLong(int filenamelenth, int limit);
	inline void EnvironBitCheck();
	//********************************************************************************************
	inline void Normlz(int n, float *x, float *a);
	inline void MatMcl(float *AB, float **A, int Arow, int Acol, float *B);
	inline void avpu(float **A, int Arow, int Acol, float *u, float *v, float alfa, float beta);
	inline void atupv(float **A, int Arow, int Acol, float *u, float *v, float alfa, float beta);
	inline void pstomo(float **A, int Arow, int Acol, float *b, int itmax);
	//*******************************************************************************
	inline bool ModelMeshCompare(MeshFrame a, MeshFrame b);
	inline bool ModelGridCompare(GridFrame a, GridFrame b);
	inline bool FrameCompare(Frame a, Frame b);
	inline char *str2charx(string a);

	inline double randu();
	namespace racc_rng
	{
		inline void set_seed_random();
	}

	inline bool Jacobi(double *matrix, int dim, double *eigenvectors, double *eigenvalues, double precision, int max);

	inline bool CheckGrid(GridFrame _grid)
	{
		if (_grid.n_rows > 1 && _grid.n_cols > 1 && _grid.n_slices > 1 &&
			_grid.d_rows > 0 && _grid.d_cols > 0 && _grid.d_slices > 0)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	inline string space_str(int n)
	{
		string str;
		for (int i = 0; i < n; i++)
			str.append(" ");
		return str;
	}
}
namespace racc_idx
{
	inline racc::IdxRange range(int _n1_l, int _n1_r, int _n2_l = 0, int _n2_r = 0, int _n3_l = 0, int _n3_r = 0)
	{
		racc::IdxRange _range;
		if (_n1_r >= _n1_l && _n2_r >= _n2_l && _n3_r >= _n3_l)
		{
			if (_n2_l == _n2_r && _n2_l == 0 && _n3_l == _n3_r && _n3_l == 0)
			{
				_range.row_l = 0;
				_range.row_r = 0;
				_range.col_l = 0;
				_range.col_r = 0;
				_range.slice_l = _n1_l;
				_range.slice_r = _n1_r;
			}
			else if (_n2_l < _n2_r && _n3_l == _n3_r && _n3_l == 0)
			{
				_range.row_l = 0;
				_range.row_r = 0;
				_range.col_l = _n1_l;
				_range.col_r = _n1_r;
				_range.slice_l = _n2_l;
				_range.slice_r = _n2_r;
			}
			else if (_n2_l < _n2_r && _n3_l < _n3_r)
			{
				_range.row_l = _n1_l;
				_range.row_r = _n1_r;
				_range.col_l = _n2_l;
				_range.col_r = _n2_r;
				_range.slice_l = _n3_l;
				_range.slice_r = _n3_r;
			}
			return _range;
		}
		else
		{
			printf("range():[ERROR]: idx range error!");
			RACC_ERROR_EXIT;
		}
	}
}
#endif