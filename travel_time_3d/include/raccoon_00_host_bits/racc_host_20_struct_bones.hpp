#pragma once
#ifndef RACC_STRUCT_BONES
#define RACC_STRUCT_BONES
//**********************************Developer******************************************
// 2020.04.10 BY CAIWEI CALVIN CAI
//*************************************************************************************
#include "racc_host_1_enum.hpp"
//*******************************STRUCR_DEFINE*****************************************
namespace racc
{
	//*******************************ENUM************************************************
	struct LineFrame
	{
		uint32_t n_rows;
		float d_rows;
		float l_rows;
		float r_rows;
		LineFrame();
		LineFrame(uint32_t _n_rows, float _d_rows);
		void copy(const LineFrame &_line);
	};
	//
	struct MeshFrame
	{
		uint32_t n_rows, n_cols;
		float d_rows, d_cols;
		float l_rows, l_cols;
		float r_rows, r_cols;
		MeshFrame();
		MeshFrame(uint32_t _n_rows, uint32_t _n_cols, float _d_rows, float _d_cols);
		void setnd(uint32_t _n_rows, uint32_t _n_cols, float _d_rows, float _d_cols);
		void copy(const MeshFrame &_mesh);
	};
	struct GridFrame
	{
		uint32_t n_rows, n_cols, n_slices;
		float d_rows, d_cols, d_slices;
		float l_rows, l_cols, l_slices;
		float r_rows, r_cols, r_slices;
		GridFrame();
		GridFrame(uint32_t _n_rows, uint32_t _n_cols, uint32_t _n_slices, float _d_rows, float _d_cols, float _d_slices);
		void setnd(uint32_t _n_rows, uint32_t _n_cols, uint32_t _n_slices, float _d_rows, float _d_cols, float _d_slices);
		void setndl(uint32_t _n_rows, uint32_t _n_cols, uint32_t _n_slices, float _d_rows, float _d_cols, float _d_slices, float _l_rows, float _l_cols, float _l_slices);
		void copy(const GridFrame &_grid);
		void PrintGridInfo(string name);
	};
	//
	struct Frame
	{
		FrameType frame_type = FrameType::null;
		uint8_t elem_byte;
		uint32_t n_rows, n_cols, n_slices;
		float d_rows, d_cols, d_slices;
		float l_rows, l_cols, l_slices;
		float r_rows, r_cols, r_slices;
		uint32_t n_elem_slice;
		uint64_t n_elem;
		//
		Frame();
		Frame sparse(uint32_t a);
		void setn(uint32_t _n_rows, uint32_t _n_cols, uint32_t _n_slices);
		void setnd(uint32_t _n_rows, float _d_rows);
		void setnd(uint32_t _n_rows, uint32_t _n_cols, float _d_rows, float _d_cols);
		void setnd(uint32_t _n_rows, uint32_t _n_cols, uint32_t _n_slices, float _d_rows, float _d_cols, float _d_slices);
		void setndl(uint32_t _n_rows, uint32_t _n_cols, uint32_t _n_slices, float _d_rows, float _d_cols, float _d_slices, float _l_rows, float _l_cols, float _l_slices);
		void copy(const Frame &_frame);
		void re_setndl();
		LineFrame ToModelLine();
		MeshFrame ToModelMesh();
		GridFrame ToModelGrid();
		void set_model_type();
		FrameType get_model_type();
		bool check();
		bool isGrid();
		bool isMesh();
		void PrintInfo(string s = "");
	};
	//
	Frame ToFrame(MeshFrame mesh);
	Frame ToFrame(GridFrame grid);
	//
	template <typename T>
	class Component2
	{
	public:
		T x;
		T y;
		enum class c
		{
			x = 0,
			y,
		};
	};
	//
	struct Coord2D : public Component2<double>
	{
		void print();
		double GetDistanceTo(Coord2D a);
	};
	//
	struct Point2D : Coord2D
	{
		uint32_t ind;
		void print();
	};
	//
	struct uvec2 : public Component2<double>
	{
		bool is_unit()
		{
			if (x * x + y * y > 0.999999)
				return true;
			else
				return false;
		}
	};
	//
	template <typename T>
	class Component3
	{
	public:
		T x;
		T y;
		T z;
		enum class c
		{
			x = 0,
			y,
			z
		};
	};
	//
	typedef Component3<float> fcomp3;
	struct Coord3D : public Component3<float>
	{
		void print();
		double GetDistanceTo(Coord3D a);
	};
	//
	struct Point3D : Coord3D
	{
		uint32_t ind;
		void print();
		void set_pos(Coord3D a);
		Coord3D get_pos();
	};
	//
	struct uvec3 : public Component3<float>
	{
		bool is_unit()
		{
			if (x * x + y * y + z * z > 0.999999)
				return true;
			else
				return false;
		}
	};
	//
	struct IdxRange
	{
		int row_l, row_r;
		int col_l, col_r;
		int slice_l, slice_r;
		IdxRange()
		{
			row_l = 0, row_r = 0;
			col_l = 0, col_r = 0;
			slice_l = 0, slice_r = 0;
		};
		IdxRange(int _row_l, int _row_r, int _col_l, int _col_r, int _slice_l, int _slice_r)
		{
			row_l = _row_l;
			row_r = _row_r;
			col_l = _col_l;
			col_r = _col_r;
			slice_l = _slice_l;
			slice_r = _slice_r;
		}
	};
}
//*************************************************************************************
#endif