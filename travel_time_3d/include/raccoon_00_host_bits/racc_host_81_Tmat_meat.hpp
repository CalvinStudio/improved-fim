#pragma once
#ifndef RACC_HOST_TMAT_MEAT
#define RACC_HOST_TMAT_MEAT
//**********************************Developer*************************************
// 2020.04.10 BY CAIWEI CALVIN CAI
//********************************************************************************
#include "racc_host_80_Tmat_bones.hpp"
//********************************CLASS_TEMPLATE**********************************

namespace racc
{
	template <typename T>
	inline Mat<T>::Mat()
	{
		row_l = 0;
		row_r = 0;
		col_l = 0;
		col_r = 0;
		n_rows = 0;
		n_cols = 0;
		n_elem = 0;
		mem = 0;
	}

	template <typename T>
	inline Mat<T>::Mat(int _n_rows, int _n_cols)
	{
		row_l = 0;
		col_l = 0;
		row_r = _n_rows - 1;
		col_r = _n_cols - 1;
		n_rows = _n_rows;
		n_cols = _n_cols;
		n_elem = n_rows * n_cols;
		mem = ralloc2d<T>(row_l, row_r, col_l, col_r);
#ifdef RACC_DEBUG
		mem_state = MemState::allocated;
#endif
	}

	template <typename T>
	inline Mat<T>::Mat(int _row_l, int _row_r, int _col_l, int _col_r)
	{
		row_l = _row_l;
		row_r = _row_r;
		col_l = _col_l;
		col_r = _col_r;
		n_rows = _row_r - _row_l + 1;
		n_cols = _col_r - _col_l + 1;
		n_elem = n_rows * n_cols;
		mem = ralloc2d<T>(row_l, row_r, col_l, col_r);
#ifdef RACC_DEBUG
		mem_state = MemState::allocated;
#endif
	}

	template <typename T>
	inline Mat<T>::Mat(int _n_rows, int _n_cols, Fill _flag)
	{
		row_l = 0;
		col_l = 0;
		row_r = _n_rows - 1;
		col_r = _n_cols - 1;
		n_rows = _n_rows;
		n_cols = _n_cols;
		n_elem = n_rows * n_cols;
		mem = ralloc2d<T>(row_l, row_r, col_l, col_r);
		_fill(_flag);
#ifdef RACC_DEBUG
		mem_state = MemState::allocated;
#endif
	}

	template <typename T>
	inline Mat<T>::Mat(const MeshFrame &_mesh)
	{
		mesh.copy(_mesh);
		this->alloc(_mesh.n_rows, _mesh.n_cols);
#ifdef RACC_DEBUG
		array_type = ArrayType::mesh;
#endif
	}

	template <typename T>
	inline void Mat<T>::alloc(const MeshFrame &_mesh)
	{
		mesh.copy(_mesh);
		this->alloc(_mesh.n_rows, _mesh.n_cols);
#ifdef RACC_DEBUG
		array_type = ArrayType::mesh;
#endif
	}

	template <typename T>
	inline void Mat<T>::alloc(int _n_rows, int _n_cols)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::inital)
		{
#endif
			row_l = 0;
			col_l = 0;
			row_r = _n_rows - 1;
			col_r = _n_cols - 1;
			n_rows = _n_rows;
			n_cols = _n_cols;
			n_elem = n_rows * n_cols;
			mem = ralloc2d<T>(row_l, row_r, col_l, col_r);
#ifdef RACC_DEBUG
			mem_state = MemState::allocated;
		}
		else if (mem_state == MemState::allocated && (_n_rows == n_rows && _n_cols == n_cols))
		{
			printf("alloc():[WARNING]:Mat memory has been allocated!\n");
		}
		else if (mem_state == MemState::allocated && !(_n_rows == n_rows && _n_cols == n_cols))
		{
			printf("alloc():[ERROR]:Mat memory has been allocated. If it needs to be changed, it needs to be released first!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline void Mat<T>::alloc(int _row_l, int _row_r, int _col_l, int _col_r)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::inital)
		{
#endif
			row_l = _row_l;
			row_r = _row_r;
			col_l = _col_l;
			col_r = _col_r;
			n_rows = _row_r - _row_l + 1;
			n_cols = _col_r - _col_l + 1;
			n_elem = n_rows * n_cols;
			mem = ralloc2d<T>(row_l, row_r, col_l, col_r);
#ifdef RACC_DEBUG
			mem_state = MemState::allocated;
		}
		else if (mem_state == MemState::allocated)
		{
			printf("alloc():[WARNING]:Mat memory has been allocated!\n");
		}
#endif
	}

	template <typename T>
	inline void Mat<T>::fill(T a)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated)
		{
#endif
			MAT_FOR2D
			mem[i][j] = a;
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("fill():[ERROR]:Mat memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("fill():[ERROR]:Mat memory has been released!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline T &Mat<T>::operator()(int i, int j)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (i >= row_l && i <= row_r) && (j >= col_l && j <= col_r))
		{
#endif
			return mem[i][j];
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("operator(index):[ERROR]:Mat memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("operator(index):[ERROR]:Mat memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(i >= row_l && i <= row_r))
		{
			printf("operator(index):[ERROR]:Mat row index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(j >= col_l && j <= col_r))
		{
			printf("operator(index):[ERROR]::Mat col index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("operator(index):[ERROR]::Mat ERROR!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline T &Mat<T>::operator()(int i, int j) const
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (i >= row_l && i <= row_r) && (j >= col_l && j <= col_r))
		{
#endif
			return mem[i][j];
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("operator(index)const:[ERROR]:Mat memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("operator(index)const:[ERROR]:Mat memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(i >= row_l && i <= row_r))
		{
			printf("operator(index)const:[ERROR]:Mat row index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(j >= col_l && j <= col_r))
		{
			printf("operator(index)const:[ERROR]::Mat col index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("operator(index)const:[ERROR]::Mat ERROR!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline Mat<T> Mat<T>::t()
	{
		Mat<T> a_t(n_cols, n_rows);
		for (int i = 0; i < n_cols; i++)
			for (int j = 0; j < n_rows; j++)
			{
				a_t(i, j) = mem[j][i];
			}
		return a_t;
	}

	template <typename T>
	inline Vec<T> Mat<T>::row(int _n_row)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (_n_row >= row_l && _n_row <= row_r))
		{
#endif
			Vec<T> a(n_cols);
			for (int i = col_l; i <= col_r; i++)
				a(i) = mem[_n_row][i];
			return a;

#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("row(idx):[ERROR]:Mat memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("row(idx):[ERROR]:Mat memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(_n_row >= row_l && _n_row <= row_r))
		{
			printf("row(idx):[ERROR]:Mat row index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("row(idx):[ERROR]::Mat ERROR!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline Mat<T> Mat<T>::rows(int l_row, int r_row)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (l_row >= row_l && l_row <= row_r) && (r_row >= row_l && r_row <= row_r) && (r_row > l_row))
		{
#endif
			Mat<T> a(r_row - l_row + 1, n_cols);
			for (int i = l_row; i <= r_row; i++)
				for (int j = 0; j <= n_cols; j++)
					a(i, j) = mem[i][j];
			return a;
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("rows(range):[ERROR]:Mat memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("rows(range):[ERROR]:Mat memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(l_row >= row_l && l_row <= row_r) || !(r_row >= row_l && r_row <= row_r))
		{
			printf("rows(range):[ERROR]:Mat row index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (r_row < l_row)
		{
			printf("rows(range):[ERROR]:Mat parameter error!\n");
			RACC_ERROR_EXIT;
		}
		else if (r_row == l_row)
		{
			printf("rows(range):[ERROR]:Mat parameter error! Please use row()!\n");
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("rows(range):[ERROR]::Mat ERROR!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline Vec<T> Mat<T>::col(int n_col)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (n_col >= col_l && n_col <= col_r))
		{
#endif
			Vec<T> a(n_rows);
			for (int i = row_l; i <= row_r; i++)
				a(i) = mem[i][n_col];
			return a;
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("col(idx):[ERROR]:Mat memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("col(idx):[ERROR]:Mat memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(n_col < col_l && n_col > col_r))
		{
			printf("col(idx):[ERROR]:Mat col index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("col(idx):[ERROR]::Mat ERROR!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline Mat<T> Mat<T>::cols(int _col_l, int _col_r)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (_col_l >= col_l && _col_l <= col_r) && (_col_r >= col_l && _col_r <= col_r) && (_col_r > _col_l))
		{
#endif
			Mat<T> a(n_rows, _col_r - _col_l + 1);
			for (int i = 0; i <= n_rows; i++)
				for (int j = _col_l; j <= _col_r; j++)
					a(i, j) = mem[i][j];
			return a;
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("cols(range):[ERROR]:Mat memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("cols(range):[ERROR]:Mat memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(_col_l >= col_l && _col_l < col_r) || !(_col_r >= col_l && _col_r < col_r))
		{
			printf("cols(range):[ERROR]:Mat col index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (_col_r < _col_l)
		{
			printf("cols(range):[ERROR]:Mat parameter error!\n");
			RACC_ERROR_EXIT;
		}
		else if (_col_r == _col_l)
		{
			printf("cols(range):[ERROR]:Mat parameter error! Please use col()!\n");
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("cols(range):[ERROR]::Mat ERROR!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline Mat<T> Mat<T>::operator()(int li, int ri, int lj, int rj)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (li >= row_l && ri <= row_r) && (lj >= col_l && rj <= col_r) && (li <= ri) && (lj <= rj))
		{
#endif
			Mat<T> a(li, ri, lj, rj);
			for (int i = li; i <= ri; i++)
				for (int j = lj; j <= rj; j++)
					a(i, j) = mem[i][j];
			return a;
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("operator(range):[ERROR]:Mat memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("operator(range):[ERROR]:Mat memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(li >= row_l && ri <= row_r))
		{
			printf("operator(range):[ERROR]:Mat row index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(lj >= col_l && rj <= col_r))
		{
			printf("operator(range):[ERROR]::Mat col index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (li > ri)
		{
			printf("operator(range):[ERROR]::Mat ROW left index can not be larger than right index!\n");
			RACC_ERROR_EXIT;
		}
		else if (lj > rj)
		{
			printf("operator(range):[ERROR]::Mat COL left index can not be larger than right index!\n");
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("operator(range):[ERROR]::Mat ERROR!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline Mat<T> Mat<T>::submat(int li, int ri, int lj, int rj)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (li >= row_l && ri <= row_r) && (lj >= col_l && rj <= col_r) && (li <= ri) && (lj <= rj))
		{
#endif
			Mat<T> a(li, ri, lj, rj);
			for (int i = li; i <= ri; i++)
				for (int j = lj; j <= rj; j++)
					a(i, j) = mem[i][j];
			return a;
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("submat(range):[ERROR]:Mat memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("submat(range):[ERROR]:Mat memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(li >= row_l && ri <= row_r))
		{
			printf("submat(range):[ERROR]:Mat row index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(lj >= col_l && rj <= col_r))
		{
			printf("submat(range):[ERROR]::Mat col index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (li > ri)
		{
			printf("submat(range):[ERROR]::Mat ROW left index can not be larger than right index!\n");
			RACC_ERROR_EXIT;
		}
		else if (lj > rj)
		{
			printf("submat(range):[ERROR]::Mat COL left index can not be larger than right index!\n");
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("submat(range):[ERROR]::Mat ERROR!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

#ifdef RACC_DEBUG
	template <typename T>
	inline racc::MemState Mat<T>::get_mem_state()
	{
		return mem_state;
	}
	template <typename T>
	inline ArrayType Mat<T>::get_array_type()
	{
		return array_type;
	}

	template <typename T>
	inline void Mat<T>::set_mem_state(MemState _mem_state)
	{
		mem_state = _mem_state;
	}

	template <typename T>
	inline void Mat<T>::print_number(string a)
	{
		_ASSERT_IS_REAL_NUMBER;
		cout << a << endl;
		matprint<T>(mem, n_rows, n_cols);
	}

	template <typename T>
	inline void Mat<T>::PrintMeshInfo()
	{
		if (array_type == ArrayType::mesh)
		{
			cout << "Mesh_X_Num:" << setw(5) << mesh.n_rows << "; INTERVAL:" << setw(6) << mesh.d_rows << "; RANGE:"
				 << "[" << mesh.l_rows << ", " << mesh.r_rows << "]" << endl;
			cout << "Mesh_Z_Num:" << setw(5) << mesh.n_cols << "; INTERVAL:" << setw(6) << mesh.d_cols << "; RANGE:"
				 << "[" << mesh.l_cols << ", " << mesh.r_cols << "]" << endl;
		}
		else
		{
			printf("PrintMeshInfo():[ERROR]:NO MESH!");
			RACC_ERROR_EXIT;
		}
	}

#endif

	template <typename T>
	inline void Mat<T>::InPutGrd(string file_path)
	{
		ReadGrdFile<T>(file_path, mesh, mem);
		n_rows = mesh.n_rows;
		n_cols = mesh.n_cols;
		row_l = 0;
		row_r = n_rows - 1;
		col_l = 0;
		col_r = n_cols - 1;
		n_elem = n_rows * n_cols;
#ifdef RACC_DEBUG
		mem_state = MemState::allocated;
		array_type = ArrayType::mesh;
#endif
	}

	template <typename T>
	inline void Mat<T>::del()
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated)
		{
#endif
			rfree2d<T>(mem, row_l, row_r, col_l, col_r);
#ifdef RACC_DEBUG
			mem_state = MemState::released;
		}
		else if (mem_state == MemState::inital)
		{
			printf("del():[WARNING]:Mat memory is not allocated and does not need to be released!\n");
		}
		else if (mem_state == MemState::released)
		{
			printf("del():[WARNING]:Mat does not need to be released again!\n");
		}
#endif
	}

	template <typename T>
	inline void Mat<T>::_save_mat(string file_path, SaveFormat _format)
	{
		int fmt;
		if (_format == SaveFormat::ascii_txt)
		{
			fmt = -1;
			file_path = file_path + ".txt";
		}
		else if (_format == SaveFormat::binary_raw)
		{
			fmt = 0;
			file_path = file_path + ".raw";
		}
		else if (_format == SaveFormat::binary_fld)
		{
			fmt = 1;
			file_path = file_path + ".rac";
		}
		else
		{
			printf("Mat::save():[ERROR]:Unsupported Format!\n");
			RACC_ERROR_EXIT;
		}
		MatOutPut<T>(file_path, mem, n_rows, n_cols, row_l, col_l, fmt);
		cout << "Save Mat File [" << file_path << "]" << endl;
	}

	template <typename T>
	inline void Mat<T>::save(string file_path, SaveFormat _format)
	{
		_ASSERT_IS_REAL_NUMBER;
		if (array_type == ArrayType::mat)
		{
			_save_mat(file_path, _format);
		}
		else if (array_type == ArrayType::mesh)
		{
			if (_format == SaveFormat::ascii_grd)
			{
				file_path = file_path + ".grd";
				MeshOutPut<T>(file_path, mem, mesh);
			}
			else if (_format == SaveFormat::ascii_xyz)
			{
				file_path = file_path + ".xyz";
				MeshOutPut<T>(file_path, mem, mesh);
			}
			else
			{
				_save_mat(file_path, _format);
			}
		}
		else
		{
			printf("Save Mat Error!");
			RACC_ERROR_EXIT;
		}
	}

	template <typename T>
	inline void Mat<T>::_fill(Fill _flag)
	{
		// if (_flag == Fill::zeros)
		// {
		// 	MAT_FOR2D
		// 	mem[i][j] = 0;
		// }
		// else if (_flag == Fill::ones)
		// {
		// 	MAT_FOR2D
		// 	mem[i][j] = 1;
		// }
		// else if (_flag == Fill::randu)
		// {
		// 	RACC_RAND_SEED
		// 	MAT_FOR2D
		// 	mem[i][j] = racc::randu();
		// }
	}
}

#endif
