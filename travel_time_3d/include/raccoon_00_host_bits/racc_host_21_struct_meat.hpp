#pragma once
#ifndef RACC_STRUCT_MEAT
#define RACC_STRUCT_MEAT
//**********************************Developer******************************************
// 2020.04.10 BY CAIWEI CALVIN CAI
//*************************************************************************************
#include "racc_host_20_struct_bones.hpp"
//*******************************STRUCR_DEFINE*****************************************
namespace racc
{
	//*******************************ENUM************************************************
	// LineFrame
	inline LineFrame::LineFrame()
	{
		n_rows = 0;
		d_rows = 0;
		l_rows = 0;
		r_rows = 0;
	}
	inline LineFrame::LineFrame(uint32_t _n_rows, float _d_rows)
	{
		n_rows = _n_rows;
		d_rows = _d_rows;
		l_rows = 0;
		r_rows = d_rows * (n_rows - 1);
	}
	inline void LineFrame::copy(const LineFrame &_line)
	{
		n_rows = _line.n_rows;
		l_rows = _line.l_rows;
		d_rows = _line.d_rows;
		r_rows = _line.r_rows;
	}
	// END LineFrame

	// MeshFrame
	inline MeshFrame::MeshFrame()
	{
		n_rows = 0;
		n_cols = 0;
		d_rows = 0;
		d_cols = 0;
		l_rows = 0;
		l_cols = 0;
		r_rows = 0;
		r_cols = 0;
	}
	inline MeshFrame::MeshFrame(uint32_t _n_rows, uint32_t _n_cols, float _d_rows, float _d_cols)
	{
		n_rows = _n_rows;
		n_cols = _n_cols;
		d_rows = _d_rows;
		d_cols = _d_cols;
		l_rows = 0;
		l_cols = 0;
		r_rows = l_rows + (n_rows - 1) * d_rows;
		r_cols = l_cols + (n_cols - 1) * d_cols;
	}
	inline void MeshFrame::setnd(uint32_t _n_rows, uint32_t _n_cols, float _d_rows, float _d_cols)
	{
		n_rows = _n_rows;
		n_cols = _n_cols;
		d_rows = _d_rows;
		d_cols = _d_cols;
		l_rows = 0;
		l_cols = 0;
		r_rows = l_rows + (n_rows - 1) * d_rows;
		r_cols = l_cols + (n_cols - 1) * d_cols;
	}
	inline void MeshFrame::copy(const MeshFrame &_mesh)
	{
		n_rows = _mesh.n_rows;
		n_cols = _mesh.n_cols;
		d_rows = _mesh.d_rows;
		d_cols = _mesh.d_cols;
		l_rows = _mesh.l_rows;
		l_cols = _mesh.l_cols;
		r_rows = _mesh.r_rows;
		r_cols = _mesh.r_cols;
	}
	// END MeshFrame

	// GridFrame
	inline GridFrame::GridFrame()
	{
		n_rows = 0;
		n_cols = 0;
		n_slices = 0;
		d_rows = 0;
		d_cols = 0;
		d_slices = 0;
		l_rows = 0;
		l_cols = 0;
		l_slices = 0;
		r_rows = 0;
		r_cols = 0;
		r_slices = 0;
	}

	inline GridFrame::GridFrame(uint32_t _n_rows, uint32_t _n_cols, uint32_t _n_slices, float _d_rows, float _d_cols, float _d_slices)
	{
		n_rows = _n_rows;
		n_cols = _n_cols;
		n_slices = _n_slices;
		d_rows = _d_rows;
		d_cols = _d_cols;
		d_slices = _d_slices;
		l_rows = 0;
		l_cols = 0;
		l_slices = 0;
		r_rows = l_rows + (n_rows - 1) * d_rows;
		r_cols = l_cols + (n_cols - 1) * d_cols;
		r_slices = l_slices + (n_slices - 1) * d_slices;
	}

	inline void GridFrame::setnd(uint32_t _n_rows, uint32_t _n_cols, uint32_t _n_slices, float _d_rows, float _d_cols, float _d_slices)
	{
		n_rows = _n_rows;
		n_cols = _n_cols;
		n_slices = _n_slices;
		d_rows = _d_rows;
		d_cols = _d_cols;
		d_slices = _d_slices;
		l_rows = 0;
		l_cols = 0;
		l_slices = 0;
		r_rows = l_rows + (n_rows - 1) * d_rows;
		r_cols = l_cols + (n_cols - 1) * d_cols;
		r_slices = l_slices + (n_slices - 1) * d_slices;
	}

	inline void GridFrame::setndl(uint32_t _n_rows, uint32_t _n_cols, uint32_t _n_slices, float _d_rows, float _d_cols, float _d_slices, float _l_rows, float _l_cols, float _l_slices)
	{
		n_rows = _n_rows;
		n_cols = _n_cols;
		n_slices = _n_slices;
		d_rows = _d_rows;
		d_cols = _d_cols;
		d_slices = _d_slices;
		l_rows = _l_rows;
		l_cols = _l_cols;
		l_slices = _l_slices;
		r_rows = _l_rows + (n_rows - 1) * d_rows;
		r_cols = _l_cols + (n_cols - 1) * d_cols;
		r_slices = _l_slices + (n_slices - 1) * d_slices;
	}

	inline void GridFrame::copy(const GridFrame &_grid)
	{
		n_rows = _grid.n_rows;
		n_cols = _grid.n_cols;
		n_slices = _grid.n_slices;
		d_rows = _grid.d_rows;
		d_cols = _grid.d_cols;
		d_slices = _grid.d_slices;
		l_rows = _grid.l_rows;
		l_cols = _grid.l_cols;
		l_slices = _grid.l_slices;
		r_rows = _grid.r_rows;
		r_cols = _grid.r_cols;
		r_slices = _grid.r_slices;
	}

	inline void GridFrame::PrintGridInfo(string name)
	{
		_PLOT_LINE;
		cout << name << endl;
		cout << "\t"
			 << "Grid_X_Num:" << setw(5) << n_rows << "; INTERVAL:" << setw(5) << d_rows << "; RANGE:"
			 << "[" << l_rows << ", " << r_rows << "]" << endl;
		cout << "\t"
			 << "Grid_Y_Num:" << setw(5) << n_cols << "; INTERVAL:" << setw(5) << d_cols << "; RANGE:"
			 << "[" << l_cols << ", " << r_cols << "]" << endl;
		cout << "\t"
			 << "Grid_Z_Num:" << setw(5) << n_slices << "; INTERVAL:" << setw(5) << d_slices << "; RANGE:"
			 << "[" << l_slices << ", " << r_slices << "]" << endl;
	}
	// END GridFrame

	// Frame
	inline Frame::Frame()
	{
		n_rows = 0;
		n_cols = 0;
		n_slices = 0;
		n_elem_slice = 0;
		n_elem = 0;
		d_rows = 0;
		d_cols = 0;
		d_slices = 0;
		l_rows = 0;
		l_cols = 0;
		l_slices = 0;
		r_rows = 0;
		r_cols = 0;
		r_slices = 0;
	}

	inline Frame Frame::sparse(uint32_t a)
	{
		Frame frame_t;
		frame_t.n_rows = (n_rows - 1) / a + 1;
		frame_t.n_cols = (n_cols - 1) / a + 1;
		frame_t.n_slices = (n_slices - 1) / a + 1;
		frame_t.n_elem_slice = frame_t.n_rows * frame_t.n_cols;
		frame_t.n_elem = frame_t.n_rows * frame_t.n_cols * frame_t.n_slices;
		frame_t.d_rows = d_rows * a;
		frame_t.d_cols = d_cols * a;
		frame_t.d_slices = d_slices * a;
		frame_t.l_rows = l_rows;
		frame_t.l_cols = l_cols;
		frame_t.l_slices = l_slices;
		frame_t.r_rows = frame_t.l_rows + (frame_t.n_rows - 1) * frame_t.d_rows;
		frame_t.r_cols = frame_t.l_cols + (frame_t.n_cols - 1) * frame_t.d_cols;
		frame_t.r_slices = frame_t.l_slices + (frame_t.n_slices - 1) * frame_t.d_slices;
		return frame_t;
	}

	inline void Frame::setnd(uint32_t _n_rows, float _d_rows)
	{
		n_rows = _n_rows;
		n_cols = 1;
		n_slices = 1;
		n_elem_slice = n_rows * n_cols;
		n_elem = n_rows * n_cols * n_slices;
		//
		d_rows = _d_rows;
		d_cols = 0;
		d_slices = 0;
		l_rows = 0;
		l_cols = 0;
		l_slices = 0;
		r_rows = l_rows + (n_rows - 1) * d_rows;
		r_cols = 0;
		r_slices = 0;
		set_model_type();
	}

	inline void Frame::setnd(uint32_t _n_rows, uint32_t _n_cols, float _d_rows, float _d_cols)
	{
		n_rows = _n_rows;
		n_cols = _n_cols;
		n_slices = 1;
		n_elem_slice = n_rows * n_cols;
		n_elem = n_rows * n_cols * n_slices;
		//
		d_rows = _d_rows;
		d_cols = _d_cols;
		d_slices = 0;
		l_rows = 0;
		l_cols = 0;
		l_slices = 0;
		r_rows = l_rows + (n_rows - 1) * d_rows;
		r_cols = l_cols + (n_cols - 1) * d_cols;
		r_slices = 0;
		set_model_type();
	}

	inline void Frame::setn(uint32_t _n_rows, uint32_t _n_cols, uint32_t _n_slices)
	{
		n_rows = _n_rows;
		n_cols = _n_cols;
		n_slices = _n_slices;
		n_elem_slice = n_rows * n_cols;
		n_elem = n_rows * n_cols * n_slices;
		//
		d_rows = 0;
		d_cols = 0;
		d_slices = 0;
		l_rows = 0;
		l_cols = 0;
		l_slices = 0;
		r_rows = 0;
		r_cols = 0;
		r_slices = 0;
		set_model_type();
	}

	inline void Frame::setnd(uint32_t _n_rows, uint32_t _n_cols, uint32_t _n_slices, float _d_rows, float _d_cols, float _d_slices)
	{
		n_rows = _n_rows;
		n_cols = _n_cols;
		n_slices = _n_slices;
		n_elem_slice = n_rows * n_cols;
		n_elem = n_rows * n_cols * n_slices;
		//
		d_rows = _d_rows;
		d_cols = _d_cols;
		d_slices = _d_slices;
		l_rows = 0;
		l_cols = 0;
		l_slices = 0;
		r_rows = l_rows + (n_rows - 1) * d_rows;
		r_cols = l_cols + (n_cols - 1) * d_cols;
		r_slices = l_slices + (n_slices - 1) * d_slices;
		set_model_type();
	}

	inline void Frame::setndl(uint32_t _n_rows, uint32_t _n_cols, uint32_t _n_slices, float _d_rows, float _d_cols, float _d_slices, float _l_rows, float _l_cols, float _l_slices)
	{
		n_rows = _n_rows;
		n_cols = _n_cols;
		n_slices = _n_slices;
		n_elem_slice = n_rows * n_cols;
		n_elem = n_rows * n_cols * n_slices;
		//
		d_rows = _d_rows;
		d_cols = _d_cols;
		d_slices = _d_slices;
		l_rows = _l_rows;
		l_cols = _l_cols;
		l_slices = _l_slices;
		if (n_rows > 0)
		{
			r_rows = l_rows + (n_rows - 1) * d_rows;
		}
		else
		{
			l_rows = r_rows = 0;
		}
		if (n_cols > 0)
		{
			r_cols = l_cols + (n_cols - 1) * d_cols;
		}
		else
		{
			l_cols = r_cols = 0;
		}
		if (n_slices > 0)
		{
			r_slices = l_slices + (n_slices - 1) * d_slices;
		}
		else
		{
			l_slices = r_slices = 0;
		}
		set_model_type();
	}

	inline void Frame::re_setndl()
	{
		r_rows = l_rows + (n_rows - 1) * d_rows;
		r_cols = l_cols + (n_cols - 1) * d_cols;
		r_slices = l_slices + (n_slices - 1) * d_slices;
		if (n_elem_slice <= 0)
			n_elem_slice = n_rows * n_cols;
		if (n_elem <= 0)
			n_elem = n_elem_slice * n_slices;
		set_model_type();
	}

	inline void Frame::copy(const Frame &_frame)
	{
		n_rows = _frame.n_rows;
		n_cols = _frame.n_cols;
		n_slices = _frame.n_slices;
		n_elem_slice = n_rows * n_cols;
		n_elem = n_elem_slice * n_slices;
		//
		d_rows = _frame.d_rows;
		d_cols = _frame.d_cols;
		d_slices = _frame.d_slices;
		l_rows = _frame.l_rows;
		l_cols = _frame.l_cols;
		l_slices = _frame.l_slices;
		r_rows = _frame.r_rows;
		r_cols = _frame.r_cols;
		r_slices = _frame.r_slices;
		frame_type = _frame.frame_type;
	}

	inline LineFrame Frame::ToModelLine()
	{
		if (frame_type == FrameType::line)
		{
			LineFrame line;
			if (n_rows > 1)
			{
				line.l_rows = l_rows;
				line.n_rows = n_rows;
				line.d_rows = d_rows;
				line.r_rows = r_rows;
			}
			else if (n_cols > 1)
			{
				line.l_rows = l_cols;
				line.n_rows = n_cols;
				line.d_rows = d_cols;
				line.r_rows = r_cols;
			}
			else if (n_slices > 1)
			{
				line.l_rows = l_slices;
				line.n_rows = n_slices;
				line.d_rows = d_slices;
				line.r_rows = r_slices;
			}
			return line;
		}
		else
		{
			printf("ToModelLine():[ERROR]:Frame cannot be converted to ModelLine!");
			RACC_ERROR_EXIT;
		}
	}

	inline MeshFrame Frame::ToModelMesh()
	{
		if (frame_type == FrameType::mesh)
		{
			MeshFrame mesh;
			if (n_slices == 1)
			{
				mesh.l_rows = l_rows;
				mesh.l_cols = l_cols;
				mesh.n_rows = n_rows;
				mesh.n_cols = n_cols;
				mesh.d_rows = d_rows;
				mesh.d_cols = d_cols;
				mesh.r_rows = r_rows;
				mesh.r_cols = r_cols;
			}
			else if (n_cols == 1)
			{
				mesh.l_rows = l_rows;
				mesh.l_cols = l_slices;
				mesh.n_rows = n_rows;
				mesh.n_cols = n_slices;
				mesh.d_rows = d_rows;
				mesh.d_cols = d_slices;
				mesh.r_rows = r_rows;
				mesh.r_cols = r_slices;
			}
			else if (n_rows == 1)
			{
				mesh.l_rows = l_cols;
				mesh.l_cols = l_slices;
				mesh.n_rows = n_cols;
				mesh.n_cols = n_slices;
				mesh.d_rows = d_cols;
				mesh.d_cols = d_slices;
				mesh.r_rows = r_cols;
				mesh.r_cols = r_slices;
			}
			return mesh;
		}
		else
		{
			printf("ToModelMesh():[ERROR]:Frame cannot be converted to ModelMesh!");
			RACC_ERROR_EXIT;
		}
	}

	inline GridFrame Frame::ToModelGrid()
	{
		if (frame_type == FrameType::grid)
		{
			GridFrame grid;
			grid.l_rows = l_rows;
			grid.l_cols = l_cols;
			grid.l_slices = l_slices;
			grid.n_rows = n_rows;
			grid.n_cols = n_cols;
			grid.n_slices = n_slices;
			grid.d_rows = d_rows;
			grid.d_cols = d_cols;
			grid.d_slices = d_slices;
			grid.r_rows = r_rows;
			grid.r_cols = r_cols;
			grid.r_slices = r_slices;
			return grid;
		}
		else
		{
			printf("ToModelGrid():[ERROR]:Frame cannot be converted to ModelGrid!");
			RACC_ERROR_EXIT;
		}
	}

	inline void Frame::set_model_type()
	{
		if (d_rows > 0 || d_cols > 0 || d_slices > 0)
		{
			if ((n_rows == 1 && n_cols == 1) || (n_rows == 1 && n_slices == 1) || (n_cols == 1 && n_slices == 1))
				frame_type = FrameType::line;
			else if ((n_rows > 1 && n_cols > 1 && n_slices == 1) || (n_rows > 1 && n_slices > 1 && n_cols == 1) || (n_cols > 1 && n_slices > 1 && n_rows == 1))
				frame_type = FrameType::mesh;
			else if (n_rows > 1 && n_cols > 1 && n_slices > 1)
				frame_type = FrameType::grid;
		}
		else
		{
			if ((n_rows == 1 && n_cols == 1) || (n_rows == 1 && n_slices == 1) || (n_cols == 1 && n_slices == 1))
				frame_type = FrameType::vec;
			else if ((n_rows > 1 && n_cols > 1 && n_slices == 1) || (n_rows > 1 && n_slices > 1 && n_cols == 1) || (n_cols > 1 && n_slices > 1 && n_rows == 1))
				frame_type = FrameType::mat;
			else if (n_rows > 1 && n_cols > 1 && n_slices > 1)
				frame_type = FrameType::cube;
		}
	}

	inline FrameType Frame::get_model_type()
	{
		return frame_type;
	}

	inline bool Frame::check()
	{
		if (frame_type == FrameType::line)
		{
			if (n_rows > 1 && n_slices == 1 && n_cols == 1)
				return true;
			else
				return false;
		}
		else if (frame_type == FrameType::mesh)
		{
			if (n_rows > 1 && n_slices > 1 && n_cols == 1)
				return true;
			else
				return false;
		}
		else
		{
			if (n_rows > 1 && n_slices > 1 && n_cols > 1)
				return true;
			else
				return false;
		}
	}

	inline bool Frame::isGrid()
	{
		if (n_rows > 1 && n_cols > 1 && n_slices > 1 &&
			d_rows > 0 && d_cols > 0 && d_slices > 0)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	inline bool Frame::isMesh()
	{
		if (n_rows > 1 && n_cols > 1 && n_slices == 1 &&
			d_rows > 0 && d_cols > 0)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	inline void Frame::PrintInfo(string s)
	{
		cout << s << endl;
		cout << "Grid_X_Num:" << setw(5) << n_rows << "; INTERVAL:" << setw(6) << d_rows << "; RANGE:"
			 << "[" << l_rows << ", " << r_rows << "]" << endl;
		cout << "Grid_Y_Num:" << setw(5) << n_cols << "; INTERVAL:" << setw(6) << d_cols << "; RANGE:"
			 << "[" << l_cols << ", " << r_cols << "]" << endl;
		cout << "Grid_Z_Num:" << setw(5) << n_slices << "; INTERVAL:" << setw(6) << d_slices << "; RANGE:"
			 << "[" << l_slices << ", " << r_slices << "]" << endl;
	}
	// END Frame

	inline Frame ToFrame(MeshFrame mesh)
	{
		Frame frame;
		frame.n_rows = mesh.n_rows;
		frame.n_cols = mesh.n_cols;
		frame.n_slices = 1;
		frame.d_rows = mesh.d_rows;
		frame.d_cols = mesh.d_cols;
		frame.d_slices = 0;
		frame.l_rows = mesh.l_rows;
		frame.l_cols = mesh.l_cols;
		frame.l_slices = 0;
		frame.r_rows = mesh.r_rows;
		frame.r_cols = mesh.r_cols;
		frame.r_slices = 0;
		frame.frame_type = FrameType::mesh;
		return frame;
	}

	inline Frame ToFrame(GridFrame grid)
	{
		Frame frame;
		frame.n_rows = grid.n_rows;
		frame.n_cols = grid.n_cols;
		frame.n_slices = grid.n_slices;
		frame.d_rows = grid.d_rows;
		frame.d_cols = grid.d_cols;
		frame.d_slices = grid.d_slices;
		frame.l_rows = grid.l_rows;
		frame.l_cols = grid.l_cols;
		frame.l_slices = grid.l_slices;
		frame.r_rows = grid.r_rows;
		frame.r_cols = grid.r_cols;
		frame.r_slices = grid.r_slices;
		frame.frame_type = FrameType::grid;
		return frame;
	}

	// Coord2D
	inline void Coord2D::print()
	{
		std::cout << "(" << x << ", " << y << ")" << std::endl;
	}

	inline double Coord2D::GetDistanceTo(Coord2D a)
	{
		return sqrt((x - a.x) * (x - a.x) + (y - a.y) * (y - a.y));
	}
	// END Coord2D

	// Coord3D
	inline void Coord3D::print()
	{
		std::cout << "(" << x << ", " << y << ", " << z << ")" << std::endl;
	}

	inline double Coord3D::GetDistanceTo(Coord3D a)
	{
		return sqrt((x - a.x) * (x - a.x) + (y - a.y) * (y - a.y) + (z - a.z) * (z - a.z));
	}
	// END Coord3D

	// Point2D
	inline void Point2D::print()
	{
		cout << "[" << ind << "]"
			 << ":(" << x << ", " << y << ")" << endl;
	}
	// END Point2D

	// Point3D
	inline void Point3D::print()
	{
		cout << "[" << ind << "]"
			 << ":(" << x << ", " << y << ", " << z << ")" << endl;
	}

	inline void Point3D::set_pos(Coord3D a)
	{
		x = a.x;
		y = a.y;
		z = a.z;
	}

	inline Coord3D Point3D::get_pos()
	{
		Coord3D a;
		a.x = x;
		a.y = y;
		a.z = z;
		return a;
	}
	// END Point3D
}
//*************************************************************************************
#endif