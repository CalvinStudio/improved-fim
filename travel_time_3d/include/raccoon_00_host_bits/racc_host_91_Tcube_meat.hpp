#pragma once
#ifndef RACC_HOST_TCUBE_MEAT
#define RACC_HOST_TCUBE_MEAT
//**********************************Developer*************************************
// 2020.04.10 BY CAIWEI CALVIN CAI
//********************************************************************************
#include "racc_host_90_Tcube_bones.hpp"
//********************************CLASS_TEMPLATE**********************************
namespace racc
{
	template <typename T>
	inline void Cube<T>::set_idx_range(int _row_l, int _row_r, int _col_l, int _col_r, int _slice_l, int _slice_r)
	{
		if (_row_l <= _row_r && _col_l <= _col_r && _slice_l <= _slice_r)
		{
			idx_range.row_l = _row_l;
			idx_range.row_r = _row_r;
			idx_range.col_l = _col_l;
			idx_range.col_r = _col_r;
			idx_range.slice_l = _slice_l;
			idx_range.slice_r = _slice_r;
		}
		else
		{
			printf("Cube::set_idx_range():[ERROR]: index error!");
			RACC_ERROR_EXIT;
		}
	}

	template <typename T>
	inline void Cube<T>::set_size(int _n_rows, int _n_cols, int _n_slices)
	{
		n_rows = _n_rows;
		n_cols = _n_cols;
		n_slices = _n_slices;
		n_slice_elem = n_rows * n_cols;
		n_elem = n_slice_elem * n_slices;
	}

	template <typename T>
	inline Cube<T>::Cube()
	{
		set_idx_range(0, 0, 0, 0, 0, 0);
		set_size(0, 0, 0);
		mem = 0;
	}
	template <typename T>
	inline Cube<T>::Cube(int _n_rows, int _n_cols, int _n_slices)
	{
		set_idx_range(0, _n_rows - 1, 0, _n_cols - 1, 0, _n_slices - 1);
		set_size(_n_rows, _n_cols, _n_slices);
		mem = ralloc3d<T>(0, _n_rows - 1, 0, _n_cols - 1, 0, _n_slices - 1);
#ifdef RACC_DEBUG
		mem_state = MemState::allocated;
#endif
	}

	template <typename T>
	inline Cube<T>::Cube(int _row_l, int _row_r, int _col_l, int _col_r, int _slice_l, int _slice_r)
	{
		set_idx_range(_row_l, _row_r, _col_l, _col_r, _slice_l, _slice_r);
		set_size(_row_r - _row_l + 1, _col_r - _col_l + 1, _slice_r - _slice_l + 1);
		mem = ralloc3d<T>(_row_l, _row_r, _col_l, _col_r, _slice_l, _slice_r);
#ifdef RACC_DEBUG
		mem_state = MemState::allocated;
#endif
	}

	template <typename T>
	inline Cube<T>::Cube(const GridFrame &_grid)
	{
		grid.copy(_grid);
		this->alloc(_grid.n_rows, _grid.n_cols, _grid.n_slices);
		array_type = ArrayType::grid;
	}

	template <typename T>
	inline void Cube<T>::alloc(int _n_rows, int _n_cols, int _n_slices)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::inital)
		{
#endif
			set_idx_range(0, _n_rows - 1, 0, _n_cols - 1, 0, _n_slices - 1);
			set_size(_n_rows, _n_cols, _n_slices);
			mem = ralloc3d<T>(0, n_rows - 1, 0, n_cols - 1, 0, n_slices - 1);
#ifdef RACC_DEBUG
			mem_state = MemState::allocated;
		}
		else if (mem_state == MemState::allocated && (_n_rows == n_rows && _n_cols == n_cols && _n_slices == n_slices))
		{
			printf("alloc():[WARNING]:Cube memory has been allocated!\n");
		}
		else if (mem_state == MemState::allocated && !(_n_rows == n_rows && _n_cols == n_cols && _n_slices == n_slices))
		{
			printf("alloc():[ERROR]:Cube memory has been allocated. If it needs to be changed, it needs to be released first!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline void Cube<T>::alloc(int _row_l, int _row_r, int _col_l, int _col_r, int _slice_l, int _slice_r)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::inital)
		{
#endif
			set_idx_range(_row_l, _row_r, _col_l, _col_r, _slice_l, _slice_r);
			set_size(_row_r - _row_l + 1, _col_r - _col_l + 1, _slice_r - _slice_l + 1);
			mem = ralloc3d<T>(_row_l, _row_r, _col_l, _col_r, _slice_l, _slice_r);
#ifdef RACC_DEBUG
			mem_state = MemState::allocated;
		}
		else if (mem_state == MemState::allocated)
		{
			printf("alloc():[WARNING]:Cube memory has been allocated!\n");
		}
#endif
	}

	template <typename T>
	inline void Cube<T>::alloc(const GridFrame &_grid)
	{
		grid.copy(_grid);
		this->alloc(_grid.n_rows, _grid.n_cols, _grid.n_slices);
		array_type = ArrayType::grid;
	}

	template <typename T>
	inline void Cube<T>::fill(T a)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated)
		{
#endif
			CUBE_FOR3D
			mem[i][j][k] = a;

#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("fill():[ERROR]:Cube memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("fill():[ERROR]:Cube memory has been released!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline T &Cube<T>::operator()(int i, int j, int k)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (i >= idx_range.row_l && i <= idx_range.row_r) && (j >= idx_range.col_l && j <= idx_range.col_r) && (k >= idx_range.slice_l && k <= idx_range.slice_r) && (mem && mem[idx_range.row_l]))
		{
#endif
			return mem[i][j][k];
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("operator(index):[ERROR]:Cube memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("operator(index):[ERROR]:Cube memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(i >= idx_range.row_l && i <= idx_range.row_r))
		{
			printf("operator(index):[ERROR]:Cube row index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(j >= idx_range.col_l && j <= idx_range.col_r))
		{
			printf("operator(index):[ERROR]::Cube col index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(k >= idx_range.slice_l && k <= idx_range.slice_r))
		{
			printf("operator(index):[ERROR]::Cube slice index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem && !mem[idx_range.row_l])
		{
			std::cout << "operator(index):[ERROR]::The memory of Cube has been released by the copied object!" << std::endl;
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("operator(index):[ERROR]:Cube ERROR!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}
	template <typename T>
	inline T &Cube<T>::operator()(int i, int j, int k) const
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (i >= idx_range.row_l && i <= idx_range.row_r) && (j >= idx_range.col_l && j <= idx_range.col_r) && (k >= idx_range.slice_l && k <= idx_range.slice_r))
		{
#endif
			return mem[i][j][k];
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("operator(index):[ERROR]:Cube memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("operator(index):[ERROR]:Cube memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(i >= idx_range.row_l && i <= idx_range.row_r))
		{
			printf("operator(index):[ERROR]:Cube row index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(j >= idx_range.col_l && j <= idx_range.col_r))
		{
			printf("operator(index):[ERROR]::Cube col index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(k >= idx_range.slice_l && k <= idx_range.slice_r))
		{
			printf("operator(index):[ERROR]::Cube slice index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("operator(index):[ERROR]:Cube ERROR!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}
	template <typename T>
	inline Cube<T> &Cube<T>::operator()(int li, int ri, int lj, int rj, int lk, int rk)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (li >= idx_range.row_l && ri <= idx_range.row_r) && (lj >= idx_range.col_l && rj <= idx_range.col_r) && (lk >= idx_range.slice_l && rk <= idx_range.slice_r) && (li <= ri) && (lj <= rj) && (lk <= rk))
		{
#endif
			Cube<T> a(li, ri, lj, rj, lk, rk);
			for (int i = li; i <= ri; i++)
				for (int j = lj; j <= rj; j++)
					for (int k = lk; k <= rk; k++)
						a(i, j, k) = mem[i][j][k];
			return a;
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("operator(range):[ERROR]:Cube memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("operator(range):[ERROR]:Cube memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(li >= idx_range.row_l && ri <= idx_range.row_r))
		{
			printf("operator(range):[ERROR]:Cube row index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(lj >= idx_range.col_l && rj <= idx_range.col_r))
		{
			printf("operator(range):[ERROR]::Cube col index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(lk >= idx_range.slice_l && rk <= idx_range.slice_r))
		{
			printf("operator(range):[ERROR]::Cube slice index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (li > ri)
		{
			printf("operator(range):[ERROR]::Cube ROW left index can not be larger than right index!\n");
			RACC_ERROR_EXIT;
		}
		else if (lj > rj)
		{
			printf("operator(range):[ERROR]::Cube COL left index can not be larger than right index!\n");
			RACC_ERROR_EXIT;
		}
		else if (lk > rk)
		{
			printf("operator(range):[ERROR]::Cube SLICE left index can not be larger than right index!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}
	template <typename T>
	inline Mat<T> Cube<T>::row(int _n_row)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (_n_row >= idx_range.row_l && _n_row <= idx_range.row_r))
		{
#endif
			Mat<T> a(n_cols, n_slices);
			for (int j = idx_range.col_l; j <= idx_range.col_r; j++)
				for (int k = idx_range.slice_l; k <= idx_range.slice_r; k++)
					a(j, k) = mem[_n_row][j][k];
			return a;
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("row(idx):[ERROR]:Cube memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("row(idx):[ERROR]:Cube memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(_n_row >= idx_range.row_l && _n_row <= idx_range.row_r))
		{
			printf("row(idx):[ERROR]:Cube row index exceeded!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline Mat<T> Cube<T>::col(int _n_col)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (_n_col >= idx_range.col_l && _n_col <= idx_range.col_r))
		{
#endif
			Mat<T> a(n_rows, n_slices);
			for (int i = idx_range.row_l; i <= idx_range.row_r; i++)
				for (int k = idx_range.slice_l; k <= idx_range.slice_r; k++)
					a(i, k) = mem[i][_n_col][k];
			return a;
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("col(idx):[ERROR]:Cube memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("col(idx):[ERROR]:Cube memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(_n_col >= idx_range.col_l && _n_col <= idx_range.col_r))
		{
			printf("col(idx):[ERROR]:Cube col index exceeded!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline Mat<T> Cube<T>::slice(int _n_slice)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (_n_slice >= idx_range.slice_l && _n_slice <= idx_range.slice_r))
		{
#endif
			Mat<T> a(n_rows, n_cols);
			for (int i = idx_range.row_l; i <= idx_range.row_r; i++)
				for (int j = idx_range.col_l; j <= idx_range.col_r; j++)
					a(i, j) = mem[i][j][_n_slice];
			return a;
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("slice(idx):[ERROR]:Cube memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("slice(idx):[ERROR]:Cube memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(_n_slice >= idx_range.slice_l && _n_slice <= idx_range.slice_r))
		{
			printf("slice(idx):[ERROR]:Cube slice index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("[ERROR]:Cube error!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline Cube<T> Cube<T>::rows(int _l_row, int _r_row)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (_l_row >= idx_range.row_l && _l_row <= idx_range.row_r) && (_l_row < _r_row))
		{
#endif
			Cube<T> a(_r_row - _l_row + 1, n_cols, n_slices);
			for (int i = _l_row; i <= _r_row; i++)
				for (int j = idx_range.col_l; j <= idx_range.col_r; j++)
					for (int k = idx_range.slice_l; k <= idx_range.slice_r; k++)
						a(i, j, k) = mem[i][j][k];
			return a;
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("rows(range):[ERROR]:Cube memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("rows(range):[ERROR]:Cube memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(_l_row >= idx_range.row_l && _r_row <= idx_range.row_r))
		{
			printf("rows(range):[ERROR]:Cube row index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (_l_row >= _r_row)
		{
			printf("rows(range):[ERROR]::Cube ROW left index can not be larger than right index!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline Cube<T> Cube<T>::cols(int _l_col, int _r_col)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (_l_col >= idx_range.col_l && _l_col <= idx_range.col_r) && (_l_col < _r_col))
		{
#endif
			Cube<T> a(n_rows, _r_col - _l_col + 1, n_slices);
			for (int i = idx_range.row_l; i <= idx_range.row_r; i++)
				for (int j = _l_col; j <= _r_col; j++)
					for (int k = idx_range.slice_l; k <= idx_range.slice_r; k++)
						a(i, j, k) = mem[i][j][k];
			return a;
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("cols(range):[ERROR]:Cube memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("cols(range):[ERROR]:Cube memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(_l_col >= idx_range.col_l && _r_col <= idx_range.col_r))
		{
			printf("cols(range):[ERROR]:Cube col index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (_l_col >= _r_col)
		{
			printf("cols(range):[ERROR]::Cube COL left index can not be larger than right index!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline Cube<T> Cube<T>::slices(int _l_slice, int _r_slice)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (_l_slice >= idx_range.slice_l && _l_slice <= idx_range.slice_r) && (_l_slice < _r_slice))
		{
#endif
			Cube<T> a(n_rows, n_cols, _r_slice - _l_slice + 1);
			for (int i = idx_range.row_l; i <= idx_range.row_r; i++)
				for (int j = idx_range.col_l; j <= idx_range.col_r; j++)
					for (int k = _l_slice; k <= _r_slice; k++)
						a(i, j, k) = mem[i][j][k];
			return a;
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("slices(range):[ERROR]:Cube memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("slices(range):[ERROR]:Cube memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(_l_slice >= idx_range.slice_l && _r_slice <= idx_range.slice_r))
		{
			printf("slices(range):[ERROR]:Cube slice index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (_l_slice >= _r_slice)
		{
			printf("slices(range):[ERROR]::Cube SLICE left index can not be larger than right index!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline Cube<T> Cube<T>::subcube(int li, int ri, int lj, int rj, int lk, int rk)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (li >= idx_range.row_l && ri <= idx_range.row_r) && (lj >= idx_range.col_l && rj <= idx_range.col_r) && (lk >= idx_range.slice_l && rk <= idx_range.slice_r) && (li <= ri) && (lj <= rj) && (lk <= rk))
		{
#endif
			Cube<T> a(li, ri, lj, rj, lk, rk);
			for (int i = li; i <= ri; i++)
				for (int j = lj; j <= rj; j++)
					for (int k = lk; k <= rk; k++)
						a(i, j, k) = mem[i][j][k];
			return a;
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("operator(range):[ERROR]:Cube memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("operator(range):[ERROR]:Cube memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(li >= idx_range.row_l && ri <= idx_range.row_r))
		{
			printf("operator(range):[ERROR]:Cube col index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(lj >= idx_range.col_l && rj <= idx_range.col_r))
		{
			printf("operator(range):[ERROR]::Cube col index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(lk >= idx_range.slice_l && rk <= idx_range.slice_r))
		{
			printf("operator(range):[ERROR]::Cube slice index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (li > ri)
		{
			printf("operator(range):[ERROR]::Cube ROW left index can not be larger than right index!\n");
			RACC_ERROR_EXIT;
		}
		else if (lj > rj)
		{
			printf("operator(range):[ERROR]::Cube COL left index can not be larger than right index!\n");
			RACC_ERROR_EXIT;
		}
		else if (lk > rk)
		{
			printf("operator(range):[ERROR]::Cube SLICE left index can not be larger than right index!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

#ifdef RACC_DEBUG

	template <typename T>
	inline MemState Cube<T>::get_mem_state()
	{
		return mem_state;
	}

	template <typename T>
	inline void Cube<T>::set_mem_state(MemState _mem_state)
	{
		mem_state = _mem_state;
	}

	template <typename T>
	inline void Cube<T>::print()
	{
		_ASSERT_IS_REAL_NUMBER;
		Mat<T> a(n_rows, n_cols);
		if (n_slices <= 6)
		{
			for (int k = idx_range.slice_l; k <= idx_range.slice_r; k++)
			{
				for (int i = idx_range.row_l; i <= idx_range.row_r; i++)
					for (int j = idx_range.col_l; j <= idx_range.col_r; j++)
						a(i, j) = mem[i][j][k];
				a.print_number();
			}
		}
		else
		{
			for (int k = idx_range.slice_l; k < 3 + idx_range.slice_l; k++)
			{
				for (int i = idx_range.row_l; i <= idx_range.row_r; i++)
					for (int j = idx_range.col_l; j <= idx_range.col_r; j++)
						a(i, j) = mem[i][j][k];
				a.print_number();
			}
			printf("\n.......................................................................\n\n");
			for (int k = idx_range.slice_r - 3; k <= idx_range.slice_r; k++)
			{
				for (int i = idx_range.row_l; i <= idx_range.row_r; i++)
					for (int j = idx_range.col_l; j <= idx_range.col_r; j++)
						a(i, j) = mem[i][j][k];
				a.print_number();
			}
		}
		a.del();
	}

	template <typename T>
	inline void Cube<T>::PrintGridInfo()
	{
		_PLOT_LINE;
		PRINT_GRID_INFO(grid);
		_PLOT_LINE;
	}

#endif

	template <typename T>
	inline void Cube<T>::save(string _path, SaveFormat _format)
	{
		_ASSERT_IS_REAL_NUMBER;

		int fmt;
		if (_format == SaveFormat::ascii_txt)
		{
			fmt = -1;
			_path = _path + ".txt";
			OutputTensor<T>(_path, mem, n_rows, n_cols, n_slices, idx_range.row_l, idx_range.col_l, idx_range.slice_l, fmt);
		}
		else if (_format == SaveFormat::binary_raw)
		{
			fmt = 0;
			_path = _path + ".raw";
			OutputTensor<T>(_path, mem, n_rows, n_cols, n_slices, idx_range.row_l, idx_range.col_l, idx_range.slice_l, fmt);
		}
		else if (_format == SaveFormat::binary_fld)
		{
			fmt = 1;
			_path = _path + ".rac";
			OutputTensor<T>(_path, mem, n_rows, n_cols, n_slices, idx_range.row_l, idx_range.col_l, idx_range.slice_l, fmt);
		}
		else if (_format == SaveFormat::ascii_xyz && array_type == ArrayType::grid)
		{
			_path = _path + ".txt";
			GridOutPut<T>(_path, mem, grid);
		}
		else
		{
			printf("Cube::save():[ERROR]:Format is error,can not save file!");
			RACC_ERROR_EXIT;
		}
		printf("Save Cube File [%s]\n", _path.c_str());
	}

	template <typename T>
	inline void Cube<T>::SliceOutPut(string file_path, string Axis, int Num)
	{
		_ASSERT_IS_REAL_NUMBER;
		if (array_type == ArrayType::grid)
		{
			file_path = file_path + ".txt";
			GridSliceOutPut(file_path, grid, mem, Axis, Num);
		}
		else
		{
			printf("Cube::GridSliceOutPut():[ERROR]:can not output grid slice!");
			RACC_ERROR_EXIT;
		}
	}

	template <typename T>
	inline void Cube<T>::del()
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated)
		{
#endif
			free3d<T>(mem, n_rows, n_cols, n_slices);
#ifdef RACC_DEBUG
			mem_state = MemState::released;
		}
		else if (mem_state == MemState::inital)
		{
			printf("del():[WARNING]:Cube memory is not allocated and does not need to be released!\n");
		}
		else if (mem_state == MemState::released)
		{
			printf("del():[WARNING]:Cube does not need to be released again!\n");
		}
#endif
	}

	template <typename T>
	inline T *Cube<T>::Trans3DTo1D()
	{
		T *array1d;
		array1d = TransArray3DTo1D<T>(mem, n_rows, n_cols, n_slices);
		return array1d;
	}
}

#endif
