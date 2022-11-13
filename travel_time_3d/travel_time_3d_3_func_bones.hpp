#pragma once
#ifndef _TRAVEL_TIME_3D_FUNC_BONES
#define _TRAVEL_TIME_3D_FUNC_BONES
#include "travel_time_3d_2_kernel_meat.hpp"
namespace jarvis
{
	inline void check_shot_line_is_out_of_bounds(Frame &model, tp3cuvec &shotline)
	{
		for (int i = 0; i < shotline.frame.n_elem; i++)
		{
			if (shotline(i).x < model.l_rows)
			{
				printf("ERROR:The shotline exceeds the model boundary!\n");
				printf("DETAILS:The %d shotpoint exceeds the model LEFT boundary!\n", i);
				std::abort();
			}
			if (shotline(i).x > model.r_rows)
			{
				printf("ERROR:The shotline exceeds the model boundary!\n");
				printf("DETAILS:The %d shotpoint exceeds the model RIGHT boundary!\n", i);
				std::abort();
			}
			if (shotline(i).y < model.l_cols)
			{
				printf("ERROR:The shotline exceeds the model boundary!\n");
				printf("DETAILS:The %d shotpoint exceeds the model FRONT boundary!\n", i);
				std::abort();
			}
			if (shotline(i).y > model.r_cols)
			{
				printf("ERROR:The shotline exceeds the model boundary!\n");
				printf("DETAILS:The %d shotpoint exceeds the model BEHIND boundary!\n", i);
				std::abort();
			}
			if (shotline(i).z < model.l_slices)
			{
				printf("ERROR:The shotline exceeds the model boundary!\n");
				printf("DETAILS:The %d shotpoint exceeds the model UP boundary!\n", i);
				std::abort();
			}
			if (shotline(i).z > model.r_slices)
			{
				printf("ERROR:The rshotline exceeds the model boundary!\n");
				printf("DETAILS:The %d shotpoint exceeds the model BOTTOM boundary!\n", i);
				std::abort();
			}
		}
	}

	inline void check_recv_line_is_out_of_bounds(Frame &model, tp3cuvec &reflectline)
	{
		for (int i = 0; i < reflectline.frame.n_elem; i++)
		{
			if (reflectline(i).x < model.l_rows)
			{
				printf("ERROR:The reflectline exceeds the model boundary!\n");
				printf("DETAILS:The %d reflectpoint exceeds the model LEFT boundary!\n", i);
				std::abort();
			}
			if (reflectline(i).x > model.r_rows)
			{
				printf("ERROR:The reflectline exceeds the model boundary!\n");
				printf("DETAILS:The %d reflectpoint exceeds the model RIGHT boundary!\n", i);
				std::abort();
			}
			if (reflectline(i).y < model.l_cols)
			{
				printf("ERROR:The reflectline exceeds the model boundary!\n");
				printf("DETAILS:The %d reflectpoint exceeds the model FRONT boundary!\n", i);
				std::abort();
			}
			if (reflectline(i).y > model.r_cols)
			{
				printf("ERROR:The reflectline exceeds the model boundary!\n");
				printf("DETAILS:The %d reflectpoint exceeds the model BEHIND boundary!\n", i);
				std::abort();
			}
			if (reflectline(i).z < model.l_slices)
			{
				printf("ERROR:The reflectline exceeds the model boundary!\n");
				printf("DETAILS:The %d reflectpoint exceeds the model UP boundary!\n", i);
				std::abort();
			}
			if (reflectline(i).z > model.r_slices)
			{
				printf("ERROR:The reflectline exceeds the model boundary!\n");
				printf("DETAILS:The %d reflectpoint exceeds the model BOTTOM boundary!\n", i);
				std::abort();
			}
		}
	}

	inline void _get_time_at(dfield &time, tp3cuvec &position) //*Solve time by interpolation
	{
		set_frame_ndl(time.frame);
		int isx, isy, isz;
		float sx, sy, sz;
		float vecl;
		int idx;
		for (int ishot = 0; ishot < position.frame.n_elem; ishot++)
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

			idx = isx + isy * time.frame.n_rows + isz * time.frame.n_elem_slice;
			float vec[3];
			if (isx <= time.frame.n_rows - 2 && isx >= 0)
			{
				vec[0] = time[idx + 1] - time[idx];
			}
			else
			{
				vec[0] = time[idx] - time[idx - 1];
			}
			//
			if (isy <= time.frame.n_cols - 2 && isy >= 0)
			{
				vec[1] = time[idx + time.frame.n_rows] - time[idx];
			}
			else
			{
				vec[1] = time[idx] - time[idx - time.frame.n_rows];
			}
			//
			if (isz <= time.frame.n_slices - 2 && isz >= 0)
			{
				vec[2] = time[idx + time.frame.n_elem_slice] - time[idx];
			}
			else
			{
				vec[2] = time[idx] - time[idx - time.frame.n_elem_slice];
			}
			sx = position(ishot).x - d_rows * isx;
			sy = position(ishot).y - d_cols * isy;
			sz = position(ishot).z - d_slices * isz;
			vecl = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
			float tdis = d_rows / vecl * vec[0];
			float ptdis = (sx * vec[0] + sy * vec[1] + sz * vec[2]) / vecl;
			position(ishot).time = time[idx] + ptdis / tdis * vec[0];
		}
	}
	inline dcufld GetFresnel(const fcufld &vel, const TimePoint3D &shot, const TimePoint3D &rece, int diff_order);
	inline dcufld GetReflectFresnel(const fcufld &vel, const TimePoint3D &shot, const TimePoint3D &rece, tp3cuvec &reflectline, int diff_order);
}
#endif