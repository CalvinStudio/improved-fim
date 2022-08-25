#pragma once
#ifndef RACC_HOST_TVEC_MEAT
#define RACC_HOST_TVEC_MEAT
//**********************************Developer*************************************
// 2020.04.10 BY CAIWEI CALVIN CAI
//********************************************************************************
#include "racc_host_70_Tvec_bones.hpp"
//********************************CLASS_TEMPLATE**********************************
namespace racc
{
	template <typename T>
	inline Vec<T>::Vec()
	{
		n_l = 0;
		n_r = 0;
		n_elem = 0;
		mem = 0;
	}

	template <typename T>
	inline Vec<T>::Vec(uint64_t _n_elem)
	{
		n_elem = _n_elem;
		n_l = 0;
		n_r = n_elem - 1;
		mem = racc::ralloc1d<T>(n_l, n_r);
#ifdef RACC_DEBUG
		mem_state = MemState::allocated;
#endif
	}

	template <typename T>
	inline Vec<T>::Vec(uint64_t _n_elem, Fill flag)
	{
		n_elem = _n_elem;
		n_l = 0;
		n_r = n_elem - 1;
		mem = racc::ralloc1d<T>(n_l, n_r);
		_fill(flag);
#ifdef RACC_DEBUG
		mem_state = MemState::allocated;
#endif
	}

	template <typename T>
	inline Vec<T>::Vec(int64_t _n_l, int64_t _n_r)
	{
		n_l = _n_l;
		n_r = _n_r;
		n_elem = _n_r - _n_l + 1;
		mem = racc::ralloc1d<T>(n_l, n_r);
#ifdef RACC_DEBUG
		mem_state = MemState::allocated;
#endif
	}

	template <typename T>
	inline void
	Vec<T>::alloc(uint64_t _n_elem)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::inital)
		{
#endif
			n_elem = _n_elem;
			n_l = 0;
			n_r = n_elem - 1;
			mem = racc::ralloc1d<T>(n_l, n_r);
#ifdef RACC_DEBUG
			mem_state = MemState::allocated;
		}
		else if (mem_state == MemState::allocated && _n_elem == n_elem)
		{
			printf("alloc():[WARNING]:Vec memory has been allocated!\n");
		}
		else if (mem_state == MemState::allocated && _n_elem != n_elem)
		{
			printf("alloc():[ERROR]:Vec memory has been allocated. If it needs to be changed, it needs to be released first!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline void
	Vec<T>::alloc(uint64_t _n_elem, Fill flag)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::inital)
		{
#endif
			n_elem = _n_elem;
			n_l = 0;
			n_r = n_elem - 1;
			mem = racc::ralloc1d<T>(n_l, n_r);
			_fill(flag);
#ifdef RACC_DEBUG
			mem_state = MemState::allocated;
		}
		else if (mem_state == MemState::allocated && _n_elem == n_elem)
		{
			printf("alloc():[WARNING]:Vec memory has been allocated!\n");
		}
		else if (mem_state == MemState::allocated && _n_elem != n_elem)
		{
			printf("alloc():[ERROR]:Vec memory has been allocated. If it needs to be changed, it needs to be released first!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline void
	Vec<T>::alloc(int64_t _n_l, int64_t _n_r)
	{
		n_l = _n_l;
		n_r = _n_r;
		n_elem = _n_r - _n_l + 1;
		mem = racc::ralloc1d<T>(n_l, n_r);
#ifdef RACC_DEBUG
		mem_state = MemState::allocated;
#endif
	}

	template <typename T>
	inline Vec<T>::Vec(const LineFrame &_line)
	{
		line.copy(_line);
		this->alloc(_line.n_rows);
		array_type = ArrayType::line;
	}

	template <typename T>
	inline void Vec<T>::alloc(const LineFrame &_line)
	{
		line.copy(_line);
		this->alloc(_line.n_rows);
		array_type = ArrayType::line;
	}

	template <typename T>
	inline void
	Vec<T>::fill(T a)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated)
		{
#endif
			for (int i = n_l; i <= n_r; i++)
			{
				mem[i] = a;
			}
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("fill():[ERROR]:Vec memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("fill():[ERROR]:Vec memory has been released!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline void
	Vec<T>::range(T a, double di, T b)
	{
#ifdef RACC_DEBUG
		_ASSERT_IS_REAL_NUMBER;
		if (mem_state == MemState::inital)
		{
			mem_state = MemState::allocated;
#endif
			n_elem = int((b - a) / (1.0 * di)) + 1;
			n_l = 0;
			n_r = n_elem - 1;
			mem = racc::ralloc1d<T>(n_l, n_r);
			for (int i = n_l; i <= n_r; i++)
			{
				mem[i] = a + di * i;
			}
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::allocated)
		{
			printf("range():[WARNING]:Vec memory has been allocated!\n");
		}
#endif
	}

	template <typename T>
	inline Vec<T> Vec<T>::subvec(int _n_l, int _n_r)
	{
#ifdef RACC_DEBUG
		if (_n_l >= this->n_l && _n_r <= this->n_r)
		{
#endif
			Vec<T> v;
			v.n_l = _n_l;
			v.n_r = _n_r;
			v.n_elem = _n_r - _n_l + 1;
			v.mem = this->mem;
#ifdef RACC_DEBUG
			v.set_mem_state(MemState::allocated);
#endif
			return v;
#ifdef RACC_DEBUG
		}
		else if (_n_l < this->n_l || _n_r > this->n_r)
		{
			printf("Vec:[ERROR]:subvec(): idx range error!");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline T &Vec<T>::operator[](int64_t i)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (i >= n_l && i <= n_r))
		{
#endif
			return mem[i];
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("operator[]:[ERROR]:Vec memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("operator[]:[ERROR]:Vec memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(i >= n_l && i <= n_r))
		{
			printf("operator[]:[ERROR]:Vec index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("operator[]:[ERROR]:Vec ERROR!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline T &Vec<T>::operator[](int64_t i) const
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (i >= n_l && i <= n_r))
		{
#endif
			return mem[i];
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("operator[]:[ERROR]:Vec memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("operator[]:[ERROR]:Vec memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(i >= n_l && i <= n_r))
		{
			printf("operator[]:[ERROR]:Vec index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("operator[]:[ERROR]:Vec ERROR!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline T &Vec<T>::operator()(int64_t i)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (i >= n_l && i <= n_r))
		{
#endif
			return mem[i];
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("operator():[ERROR]:Vec memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("operator():[ERROR]:Vec memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(i >= n_l && i <= n_r))
		{
			printf("operator():[ERROR]:Vec index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("operator():[ERROR]:Vec ERROR!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline T &Vec<T>::operator()(int64_t i) const
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (i >= n_l && i <= n_r))
		{
#endif
			return mem[i];
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("operator()const:[ERROR]:Vec memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("operator()const:[ERROR]:Vec memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (!(i >= n_l && i <= n_r))
		{
			printf("operator()const:[ERROR]:Vec index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("operator()const:[ERROR]:Vec ERROR!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline Vec<T> &Vec<T>::operator()(int li, int ri)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && (li <= ri) && (li >= n_l && ri <= n_r))
		{
#endif
			Vec<T> a(ri - li + 1);
			for (int i = li; i <= ri; i++)
			{
				a(i - li) = mem[i];
			}
			return a;
#ifdef RACC_DEBUG
		}
		else if (mem_state == MemState::inital)
		{
			printf("operator(range):[ERROR]:Vec memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
		else if (mem_state == MemState::released)
		{
			printf("operator(range):[ERROR]:Vec memory has been released!\n");
			RACC_ERROR_EXIT;
		}
		else if (li < 0)
		{
			printf("operator(range):[ERROR]:Vec left index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (ri >= n_elem)
		{
			printf("operator(range):[ERROR]:Vec right index exceeded!\n");
			RACC_ERROR_EXIT;
		}
		else if (li > ri)
		{
			printf("operator(range):[ERROR]:Vec left index can not be larger than right index!\n");
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("operator(range):[ERROR]:Vec ERROR!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline void Vec<T>::del()
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated)
		{
#endif
			rfree1d<T>(mem, n_l, n_r);
#ifdef RACC_DEBUG
			mem_state = MemState::released;
		}
		else if (mem_state == MemState::inital)
		{
			printf("del():[WARNING]:Vec memory is not allocated and does not need to be released!\n");
		}
		else if (mem_state == MemState::released)
		{
			printf("del():[WARNING]:Vec does not need to be released again!\n");
		}
#endif
	}
	//-----------------------------------------------------------------------------------------
	template <typename T>
	inline T Vec<T>::p4space(GridFrame grid, int i, int j, int k)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && n_elem == int(grid.n_rows * grid.n_cols * grid.n_slices))
#endif
			return mem[GRID_IDX3D(grid)];
#ifdef RACC_DEBUG
		else if (n_elem != int(grid.n_rows * grid.n_cols * grid.n_slices))
		{
			printf("p4space(Grid,index):[ERROR]:Grid mismatch!\n");
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("p4space(Grid,index):[ERROR]:Vec memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline T Vec<T>::p4gms(GridFrame grid, int i, int j, int k)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && n_elem == int(grid.n_rows * grid.n_cols * grid.n_slices))
#endif
			return mem[GMS_IDX3D(grid)];
#ifdef RACC_DEBUG
		else if (n_elem != int(grid.n_rows * grid.n_cols * grid.n_slices))
		{
			printf("p4gms(Grid,index):[ERROR]:Grid mismatch!\n");
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("p4gms(Grid,index):[ERROR]:Vec memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline T Vec<T>::p(MeshFrame mesh, int i, int j)
	{
#ifdef RACC_DEBUG
		if (mem_state == MemState::allocated && n_elem == int(mesh.n_rows * mesh.n_cols))
		{
#endif
			return mem[i + j * mesh.n_rows];
#ifdef RACC_DEBUG
		}
		else if (n_elem != int(mesh.n_rows * mesh.n_cols))
		{
			printf("p(Mesh,index):[ERROR]:Mesh mismatch!\n");
			RACC_ERROR_EXIT;
		}
		else
		{
			printf("p(Grid,index):[ERROR]:Vec memory not allocated!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	template <typename T>
	inline T ***Vec<T>::trans1to3(const GridFrame &grid)
	{
		if (n_elem == int(grid.n_rows * grid.n_cols * grid.n_slices))
		{
			T ***Array3D = alloc3d<T>(grid.n_rows, grid.n_cols, grid.n_slices);
			GRID_FOR_IDX3D(grid)
			Array3D[i][j][k] = mem->p(grid, i, j, k);
			return Array3D;
		}
		else
		{
			printf("An error occurred in converting 1D to 3D array due to inconsistent array size!\n");
			RACC_ERROR_EXIT;
		}
	}

	template <typename T>
	inline int Vec<T>::index_min()
	{
		int min_idx, i;
		double min = mem[n_l];
		for (i = n_l; i <= n_r; i++)
		{
			if (min >= mem[i])
			{
				min = mem[i];
				min_idx = i;
			}
		}
		return min_idx;
	}

	template <typename T>
	inline void Vec<T>::_fill(Fill flag)
	{
		// if (flag == Fill::zeros)
		// {
		// 	for (int i = 0; i < n_elem; i++)
		// 	{
		// 		mem[i] = 0;
		// 	}
		// }
		// else if (flag == Fill::ones)
		// {
		// 	for (int i = 0; i < n_elem; i++)
		// 	{
		// 		mem[i] = 1;
		// 	}
		// }
		// else if (flag == Fill::randu)
		// {
		// 	RACC_RAND_SEED
		// 	for (int i = 0; i < n_elem; i++)
		// 	{
		// 		mem[i] = racc::randu();
		// 	}
		// }
	}

	template <typename T>
	inline void Vec<T>::save(string _path, SaveFormat _format)
	{
		int _fmt;
		if (_format == SaveFormat::ascii_txt)
			_fmt = -1, _path = _path + ".txt";
		if (_format == SaveFormat::binary_raw)
			_fmt = 0, _path = _path + ".raw";
		if (_format == SaveFormat::binary_fld)
			_fmt = 1, _path = _path + ".rac";
		OutputVector(_path, mem, n_elem, n_l, _fmt);
		cout << "Save Vec File [" << _path << "]" << endl;
	}

	template <typename T>
	inline void Vec<T>::print_number(string a)
	{
#ifdef RACC_DEBUG
		cout << a << endl;
		vecprint_number<T>(n_l, n_r, mem);
#endif
	}

	template <typename T>
	inline void Vec<T>::print_struct(string a)
	{
#ifdef RACC_DEBUG
		cout << a << endl;
		vecprint_struct<T>(n_l, n_r, mem);
#endif
	}

#ifdef RACC_DEBUG
	template <typename T>
	inline MemState Vec<T>::get_mem_state() const
	{
		return mem_state;
	}

	template <typename T>
	inline void Vec<T>::set_mem_state(MemState _mem_state)
	{
		mem_state = _mem_state;
	}
#endif

#ifdef RACC_USE_ARMA
	template <typename T>
	arma::Row<T> Vec<T>::ToArmaRow()
	{
		arma::Row<T> a(n_elem);
		for (int i = 0; i < n_elem; i++)
			a(i) = mem[i];
		return a;
	}

	template <typename T>
	arma::Col<T> Vec<T>::ToArmaCol()
	{
		arma::Col<T> a(n_elem);
		for (int i = 0; i < n_elem; i++)
			a(i) = mem[i];
		return a;
	}
#endif
}

#endif
