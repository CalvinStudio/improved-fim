#pragma once
#ifndef RACC_HOST_TMAT_BONES
#define RACC_HOST_TMAT_BONES
//**********************************Developer*************************************
// 2020.04.10 BY CAIWEI CALVIN CAI
//********************************************************************************
#include "racc_host_72_Tvec_function.hpp"
//********************************CLASS_TEMPLATE**********************************

namespace racc
{
	template <typename T>
	class Mat
	{
		/**
		@param n_elem Number of matrix
		@param n_rows Number of matrix rows
		@param n_cols Number of matrix columns
		@param mem 2d array pointer
		@param mem_state(for debug) The state of memory
		*/
#ifdef RACC_DEBUG
	protected:
		MemState mem_state = MemState::inital;
#endif
	protected:
		ArrayType array_type = ArrayType::mat;

	public:
		MeshFrame mesh;
		int n_elem;
		int n_rows, n_cols;
		int row_l, row_r;
		int col_l, col_r;
		T **mem;

	public:
		Mat();
		Mat(int _n_rows, int _n_cols);
		Mat(int _row_l, int _row_r, int _col_l, int _col_r);
		Mat(int _n_rows, int _n_cols, Fill _flag);
		void alloc(int _n_rows, int _n_cols);
		void alloc(int _row_l, int _row_r, int _col_l, int _col_r);
		Mat(const MeshFrame &_mesh);
		void alloc(const MeshFrame &_mesh);
		void fill(T a);
		T &operator()(int i, int j);
		T &operator()(int i, int j) const;
		Mat<T> t();
		//!Extra memory overhead, use with caution!
		Vec<T> row(int _n_row);
		Mat<T> rows(int l_row, int r_row);
		Vec<T> col(int n_col);
		Mat<T> cols(int l_col, int r_col);
		Mat<T> operator()(int li, int ri, int lj, int rj);
		Mat<T> submat(int li, int ri, int lj, int rj);
		//!END
		void del();
		void save(string file_path, SaveFormat _format);
		void InPutGrd(string file_path);

	private:
		void _fill(Fill _flag);
		void _save_mat(string file_path, SaveFormat _format);

#ifdef RACC_DEBUG
	public:
		MemState get_mem_state();
		void set_mem_state(MemState _mem_state);
		ArrayType get_array_type();
		void print_number(string a = "Mat:");
		void PrintMeshInfo();
#endif
	};
	//******************************************************************
	typedef Mat<cx_double> cx_mat;
	typedef Mat<cx_double> cx_dmat;
	typedef Mat<cx_float> cx_fmat;
	typedef Mat<double> mat;
	typedef Mat<double> dmat;
	typedef Mat<float> fmat;
	typedef Mat<int> imat;
	//******************************************************************
}

#endif
