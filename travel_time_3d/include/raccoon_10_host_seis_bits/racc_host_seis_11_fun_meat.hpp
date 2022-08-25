#pragma once
#ifndef RACC_HOST_SEIS_FUN_MEAT
#define RACC_HOST_SEIS_FUN_MEAT
#include "racc_host_seis_10_fun_bones.hpp"
namespace racc
{
	inline sp2vec ReadSRLine2D(std::string pathname)
	{
		std::ifstream infile;
		infile.open(pathname);
#ifdef RACC_DEBUG
		if (infile.is_open())
		{
#endif
			sp2vec data;
			infile >> data.n_elem;
			data.alloc(data.n_elem);
			for (int i = 0; i < data.n_elem; i++)
			{
				infile >> data(i).x;
				infile >> data(i).y;
				data(i).ind = i;
			}
			infile.close();
			return data;
#ifdef RACC_DEBUG
		}
		else
		{
			printf("ReadSRLine2D(path):File open error!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	inline sp3vec ReadSRLine3D(std::string pathname)
	{
		std::ifstream infile;
		infile.open(pathname);
#ifdef RACC_DEBUG
		if (infile.is_open())
		{
#endif
			sp3vec data;
			infile >> data.n_elem;
			data.alloc(data.n_elem);
			for (int i = 0; i < data.n_elem; i++)
			{
				infile >> data(i).x;
				infile >> data(i).y;
				infile >> data(i).z;
				data(i).ind = i;
			}
			infile.close();
			return data;
#ifdef RACC_DEBUG
		}
		else
		{
			printf("ReadSRLine3D(path):File open error!\n");
			RACC_ERROR_EXIT;
		}
#endif
	}

	inline void CheckShotLine(Frame model, tp3vec shotline)
	{
		for (int i = 0; i < shotline.n_elem; i++)
		{
			if (shotline(i).x < model.l_rows)
			{
				_PLOT_ERROR_LINE
				printf("ERROR:The shotline exceeds the model boundary!\n");
				printf("DETAILS:The %d shotpoint exceeds the model LEFT boundary!\n", i);
				RACC_ERROR_EXIT;
			}
			if (shotline(i).x > model.r_rows)
			{
				_PLOT_ERROR_LINE
				printf("ERROR:The shotline exceeds the model boundary!\n");
				printf("DETAILS:The %d shotpoint exceeds the model RIGHT boundary!\n", i);
				RACC_ERROR_EXIT;
			}
			if (shotline(i).y < model.l_cols)
			{
				_PLOT_ERROR_LINE
				printf("ERROR:The shotline exceeds the model boundary!\n");
				printf("DETAILS:The %d shotpoint exceeds the model FRONT boundary!\n", i);
				RACC_ERROR_EXIT;
			}
			if (shotline(i).y > model.r_cols)
			{
				_PLOT_ERROR_LINE
				printf("ERROR:The shotline exceeds the model boundary!\n");
				printf("DETAILS:The %d shotpoint exceeds the model BEHIND boundary!\n", i);
				RACC_ERROR_EXIT;
			}
			if (shotline(i).z < model.l_slices)
			{
				_PLOT_ERROR_LINE
				printf("ERROR:The shotline exceeds the model boundary!\n");
				printf("DETAILS:The %d shotpoint exceeds the model UP boundary!\n", i);
				RACC_ERROR_EXIT;
			}
			if (shotline(i).z > model.r_slices)
			{
				_PLOT_ERROR_LINE
				printf("ERROR:The rshotline exceeds the model boundary!\n");
				printf("DETAILS:The %d shotpoint exceeds the model BOTTOM boundary!\n", i);
				RACC_ERROR_EXIT;
			}
		}
	}

	inline void CheckReflectLine(Frame model, tp3vec reflectline)
	{
		for (int i = 0; i < reflectline.n_elem; i++)
		{
			if (reflectline(i).x < model.l_rows)
			{
				_PLOT_ERROR_LINE
				printf("ERROR:The reflectline exceeds the model boundary!\n");
				printf("DETAILS:The %d reflectpoint exceeds the model LEFT boundary!\n", i);
				RACC_ERROR_EXIT;
			}
			if (reflectline(i).x > model.r_rows)
			{
				_PLOT_ERROR_LINE
				printf("ERROR:The reflectline exceeds the model boundary!\n");
				printf("DETAILS:The %d reflectpoint exceeds the model RIGHT boundary!\n", i);
				RACC_ERROR_EXIT;
			}
			if (reflectline(i).y < model.l_cols)
			{
				_PLOT_ERROR_LINE
				printf("ERROR:The reflectline exceeds the model boundary!\n");
				printf("DETAILS:The %d reflectpoint exceeds the model FRONT boundary!\n", i);
				RACC_ERROR_EXIT;
			}
			if (reflectline(i).y > model.r_cols)
			{
				_PLOT_ERROR_LINE
				printf("ERROR:The reflectline exceeds the model boundary!\n");
				printf("DETAILS:The %d reflectpoint exceeds the model BEHIND boundary!\n", i);
				RACC_ERROR_EXIT;
			}
			if (reflectline(i).z < model.l_slices)
			{
				_PLOT_ERROR_LINE
				printf("ERROR:The reflectline exceeds the model boundary!\n");
				printf("DETAILS:The %d reflectpoint exceeds the model UP boundary!\n", i);
				RACC_ERROR_EXIT;
			}
			if (reflectline(i).z > model.r_slices)
			{
				_PLOT_ERROR_LINE
				printf("ERROR:The reflectline exceeds the model boundary!\n");
				printf("DETAILS:The %d reflectpoint exceeds the model BOTTOM boundary!\n", i);
				RACC_ERROR_EXIT;
			}
		}
	}

	inline void _GetTimeAt(dfield &time, tp3vec &position) //*Solve time by interpolation
	{
		SET_MODEL_GRID_NDL(time.frame);
		int isx, isy, isz;
		double sx, sy, sz;
		double vecl;
		for (int ishot = 0; ishot < position.n_elem; ishot++)
		{
			isx = int((position(ishot).x - l_rows) / d_rows);
			isy = int((position(ishot).y - l_cols) / d_cols);
			isz = int((position(ishot).z - l_slices) / d_slices);
			if (isx == n_rows)
				isx = n_rows - 1;
			if (isy == n_cols)
				isy = n_cols - 1;
			if (isz == n_slices)
				isz = n_slices - 1;
			if (isx >= n_rows - 1)
				isx = n_rows - 2;

			dvec vec(3);
			vec(0) = time(isx + 1, isy, isz) - time(isx, isy, isz);
			vec(1) = time(isx, isy + 1, isz) - time(isx, isy, isz);
			vec(2) = time(isx, isy, isz + 1) - time(isx, isy, isz);
			sx = position(ishot).x - d_rows * isx;
			sy = position(ishot).y - d_cols * isy;
			sz = position(ishot).z - d_slices * isz;
			vecl = sqrt(vec(0) * vec(0) + vec(1) * vec(1) + vec(2) * vec(2));
			double tdis = d_rows / vecl * vec(0);
			double ptdis = (sx * vec(0) + sy * vec(1) + sz * vec(2)) / vecl;
			position(ishot).time = time(isx, isy, isz) + ptdis / tdis * vec(0);
		}
	}
}
#endif