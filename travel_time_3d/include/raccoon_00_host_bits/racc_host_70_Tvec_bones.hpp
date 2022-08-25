#ifndef RACC_HOST_TVEC_BONES
#define RACC_HOST_TVEC_BONES
//**********************************Developer*************************************
// 2020.04.10 BY CAIWEI CALVIN CAI
//********************************************************************************
#include "racc_host_61_Tio_meat.hpp"
//********************************CLASS_TEMPLATE**********************************
namespace racc
{
	template <typename T>
	class Vec
	{
		/**
		@param n_elem Number of vector elements
		@param n_l the left index of Vec
		@param n_r the right index of Vec
		@param mem 1d array pointer
		@param mem_state(for debug) The state of memory
		*/
#ifdef RACC_DEBUG
	protected:
		MemState mem_state = MemState::inital;
#endif
	protected:
		ArrayType array_type = ArrayType::vec;

	public:
		LineFrame line;
		uint64_t n_elem;
		int64_t n_l;
		int64_t n_r;
		T *mem;

	public:
		Vec();
		Vec(uint64_t _n_elem);
		Vec(int64_t _n_l, int64_t _n_r);
		Vec(uint64_t _n_elem, Fill flag);
		Vec(const LineFrame &_line);
		void alloc(uint64_t _n_elem);
		void alloc(int64_t _n_l, int64_t _n_r);
		void alloc(uint64_t _n_elem, Fill flag);
		void alloc(const LineFrame &_line);
		void fill(T a);
		void range(T a, double di, T b);
		Vec<T> subvec(int _n_l, int _n_r);
		T &operator[](int64_t i);
		T &operator[](int64_t i) const;
		T &operator()(int64_t i);
		T &operator()(int64_t i) const;
		Vec<T> &operator()(int li, int ri);
		T p4space(GridFrame grid, int i, int j, int k);
		T p4gms(GridFrame grid, int i, int j, int k);
		T p(MeshFrame mesh, int i, int j);
		T ***trans1to3(const GridFrame &grid);
		int index_min();
		void save(string _path, SaveFormat _format = SaveFormat::ascii_txt);
		void del();
		void print_number(string str = "Vec(num):");
		void print_struct(string str = "Vec(obj):");

	private:
		void _fill(Fill flag);

	public:
#ifdef RACC_DEBUG
		MemState get_mem_state() const;
		void set_mem_state(MemState _mem_state);
#endif

#ifdef RACC_USE_ARMA
		arma::Row<T> ToArmaRow();
		arma::Col<T> ToArmaCol();
#endif
	};
	//******************************************************************
	typedef Vec<cx_double> cx_vec;
	typedef Vec<cx_double> cx_dvec;
	typedef Vec<cx_float> cx_fvec;
	typedef Vec<double> vec;
	typedef Vec<double> dvec;
	typedef Vec<float> fvec;
	typedef Vec<int> ivec;
	//******************************************************************
	typedef Vec<Point2D> sp2vec;
	typedef Vec<Point3D> sp3vec;
	//******************************************************************
}

#endif
