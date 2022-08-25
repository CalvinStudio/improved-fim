#pragma once
#ifndef RACC_HOST_TFIELD_BONES
#define RACC_HOST_TFIELD_BONES
//**********************************Developer*************************************
// 2021.04.9 BY CAIWEI CALVIN CAI
//********************************************************************************
#include "racc_host_92_Tcube_function.hpp"
//********************************CLASS_TEMPLATE**********************************
//********************************************************************************
namespace racc
{
	template <typename T>
	class Field
	{
		/**
		@param n_elem Number of field element
		@param n_rows Number of field rows
		@param n_cols Number of field columns
		@param n_slices Number of field slices
		@param mem array pointer
		@param mem_state(for debug) The state of memory
		@param field_type(for debug) The type of field
		*/
	protected:
		T *mem;
		//
	public:
		string name = "un_named";
		Frame frame;
		//
		// construct func
		Field();
		Field(string _name);
		Field(T *_mem_ptr); /*(for fortran mixed coding)*/
		Field(uint32_t _n_rows, uint32_t _n_cols = 1, uint32_t _n_slices = 1);
		Field(const Frame &_frame);
		//
		// operator func
		T &operator[](uint64_t i);
		T &operator()(uint32_t i, uint32_t j = 0, uint32_t k = 0);
		T &operator()(uint32_t i, uint32_t j = 0, uint32_t k = 0) const;
		//
		// memory func
		void alloc(uint32_t _n_rows, uint32_t _n_cols = 1, uint32_t _n_slices = 1);
		void alloc(const Frame &_frame);
		void fill(T a);
		void zeros(uint32_t _n_rows, uint32_t _n_cols = 1, uint32_t _n_slices = 1); // alloc()+fill(0)
		void copy(const Field<T> &obj);
		void del();
		//
		// name
		void set_name(string name);
		string get_name();
		//
		// math func
		T max();
		T min();
		T abs_sum();
		T mean();
		T absmean();
		T *get_mem_p();
		T *&mem_p()
		{
			return mem;
		}
		//
		// interface func
		Vec<T> to_vec();
		Mat<T> to_mat();
		Cube<T> to_cube();
		//
		// io func
		void save(string file_path, SaveFormat _format);
		void read_binary(string file_path);
		string size_str();
		//
	protected:
		void output_binary_raw(string file_path);
		void output_binary_fld(string file_path);
		void output_ascii_xyz(string file_path);
		void output_ascii_grd(string file_path);
		void output_sgy(string file_path);
		void output_ascii_txt_2d(string file_path);
		void output_ascii_txt_3d(string file_path);
		//
#ifdef RACC_DEBUG
	protected:
		MemState mem_state = MemState::inital;
		//
	public:
#ifdef RACC_MEMORY_RELEASE_MANAGEMENT
		~Field();
#endif
		MemState get_mem_state() const;
		void set_mem_state(MemState _mem_state);
		void print_frame_info();
#endif
	};
	//******************************************************************
	typedef Field<cx_double> cx_field;
	typedef Field<cx_double> cx_dfield;
	typedef Field<cx_float> cx_ffield;
	typedef Field<double> field;
	typedef Field<double> dfield;
	typedef Field<float> ffield;
	typedef Field<int> ifield;
	//******************************************************************
}

#endif
