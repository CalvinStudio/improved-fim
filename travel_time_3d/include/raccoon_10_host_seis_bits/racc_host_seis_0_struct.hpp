#pragma once
#ifndef RACC_HOST_SEIS_STRUCT
#define RACC_HOST_SEIS_STRUCT
#include "_racc_host_seis_header_in.h"
namespace racc
{
	struct TimePoint2D : Coord2D
	{
		double time;
#ifdef RACC_DEBUG
		void print()
		{
			std::cout.setf(std::ios::left);
			std::cout << "TimePoint2D:(" << setw(6) << x << setw(6) << y << ")"
					  << ", " << time << std::endl;
		}
#endif
		void set_pos(Coord2D a)
		{
			x = a.x;
			y = a.y;
		}
		Coord2D get_pos()
		{
			Coord2D a;
			a.x = x;
			a.y = y;
			return a;
		}
	};

	struct TimePoint3D : Coord3D
	{
		double time = 0;
#ifdef RACC_DEBUG
		void print()
		{
			printf("TimePoint3D:(%6.4f, %6.4f, %6.4f): %10.6f\n", x, y, z, time);
		}
#endif
		void set_pos(const Coord3D &a)
		{
			x = a.x;
			y = a.y;
			z = a.z;
		}
		Coord3D get_pos()
		{
			Coord3D a;
			a.x = x;
			a.y = y;
			a.z = z;
			return a;
		}
	};
	struct SBackPoints
	{
		int x;
		int z;
		double time;
	};
	struct SMigPara
	{
		float x;
		float y;
		float z;
		float linelength;
		float cosfai;
		float v_average;
		float time;
		float stime;
		float rtime;
	};

	struct SMigResult
	{
		float x;
		float y;
		float z;
		float value;
	};

	struct RayLength
	{
		float rlength;
		int ix;
		int jy;
		int kz;
	};

	struct SLineDirection3D
	{
		float theta; // azimuth angle, 从X正向转一个锐角到Y正向从零增加到90度,
					 // 角度范围(0, 360)
		float phi;	 // dip angle, 从Z轴正向开始往Z轴负方向转动, 角度由零度增加到180度
		double cosx; // directional cosine value
		double cosy;
		double cosz;
		double polarization;
	};

	struct SegyFileName3C
	{
		char xc[FILE_NAME_SIZE_MAX];
		char yc[FILE_NAME_SIZE_MAX];
		char zc[FILE_NAME_SIZE_MAX];
	};

	struct ExtremeAmp3C
	{
		float maxx;
		float maxy;
		float maxz;
		float minx;
		float miny;
		float minz;
	};
	//******************************************************************
	typedef TimePoint2D tp2;
	typedef TimePoint3D tp3;
	typedef Vec<TimePoint2D> tp2vec;
	typedef Vec<TimePoint3D> tp3vec;
	typedef Vec<SBackPoints> bpvec;
	typedef Mat<SLineDirection3D> ld3mat;
	//******************************************************************
}

#endif