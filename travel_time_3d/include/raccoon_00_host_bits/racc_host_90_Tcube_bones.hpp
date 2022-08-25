#pragma once
#ifndef RACC_HOST_TCUBE_BONES
#define RACC_HOST_TCUBE_BONES
//**********************************Developer*************************************
// 2020.04.10 BY CAIWEI CALVIN CAI
//********************************************************************************
#include "racc_host_82_Tmat_function.hpp"
//********************************CLASS_TEMPLATE**********************************
namespace racc
{
	template <typename T>
	class Cube
	{
		/**
		@param n_rows Number of cube rows
		@param n_cols Number of cube columns
		@param n_slices Number of cube slices
		@param mem 3d array pointer
		@param mem_state The state of memory
		*/
#ifdef RACC_DEBUG
	protected:
		MemState mem_state = MemState::inital;
#endif
	protected:
		ArrayType array_type = ArrayType::cube;

	public:
		GridFrame grid;
		int n_rows, n_cols, n_slices;
		int n_slice_elem;
		int n_elem;
		IdxRange idx_range;
		T ***mem;

	public:
		Cube();
		Cube(int _n_rows, int _n_cols, int _n_slices);
		Cube(int _row_l, int _row_r, int _col_l, int _col_r, int _slice_l, int _slice_r);
		void alloc(int _n_rows, int _n_cols, int _n_slices);
		void alloc(int _row_l, int _row_r, int _col_l, int _col_r, int _slice_l, int _slice_r);
		Cube(const GridFrame &_grid);
		void alloc(const GridFrame &_grid);
		void fill(T a);
		void copy(const Cube<T> &obj);
		T &operator()(int i, int j, int k);
		T &operator()(int i, int j, int k) const;
		//*Extra memory overhead, use with caution!
		Cube<T> &operator()(int li, int ri, int lj, int rj, int lk, int rk);
		Mat<T> row(int _n_row);
		Mat<T> col(int _n_col);
		Mat<T> slice(int _n_slice);
		Cube<T> rows(int _l_row, int _r_row);
		Cube<T> cols(int _l_col, int _r_col);
		Cube<T> slices(int _l_slice, int _r_slice);
		Cube<T> subcube(int li, int ri, int lj, int rj, int lk, int rk);
		//! END
		void SliceOutPut(string file_path, string Axis, int Num);
		void save(string file_path, SaveFormat _format);
		void del();
		T *Trans3DTo1D();

#ifdef RACC_DEBUG
	public:
		MemState get_mem_state();
		void set_mem_state(MemState _mem_state);
		void print();
		void PrintGridInfo();
#endif
	protected:
		void set_idx_range(int _row_l, int _row_r, int _col_l, int _col_r, int _slice_l, int _slice_r);
		void set_size(int _n_rows, int _n_cols, int _n_slices);
	};
	//******************************************************************
	typedef Cube<cx_double> cx_cube;
	typedef Cube<cx_double> cx_dcube;
	typedef Cube<cx_float> cx_fcube;
	typedef Cube<double> cube;
	typedef Cube<double> dcube;
	typedef Cube<float> fcube;
	typedef Cube<int> icube;
	//******************************************************************
}

#endif
