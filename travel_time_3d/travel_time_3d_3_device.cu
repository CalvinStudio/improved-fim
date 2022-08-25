#pragma once
#ifndef GPU_TRAVEL_TIME_3D_CU
#define GPU_TRAVEL_TIME_3D_CU
#include "travel_time_3d_0.hpp"
#include "travel_time_3d_2_kernel_meat.hpp"
namespace racc
{
	void travel_time_3d_module::cu_cal_time()
	{
		// int Nmax = racc::max(n_rows, n_cols, n_slices);
		// for (int k = 0; k < Nmax * 4; k++)
		// {
		// 	fim_kernel<<<Blocks, Threads>>>(vel.frame, vel.cu_mem, time.cu_mem, mark.cu_mem,
		// 									diff_order);
		// 	cudaDeviceSynchronize();
		// }
		int count = 0;
		do
		{
			endflag[0] = false;
			cudaDeviceSynchronize();
			endflag.cu_stream_copy_h2d(racc_default_stream);
			cudaDeviceSynchronize();
			fim_kernel_0<<<Blocks, Threads>>>(vel.frame, vel.cu_mem, time.cu_mem, mark.cu_mem, diff_order);
			fim_kernel_1<<<Blocks, Threads>>>(vel.frame, vel.cu_mem, time.cu_mem, mark.cu_mem, diff_order);
			// fim_kernel_4<<<Blocks, Threads>>>(vel.frame, vel.cu_mem, time.cu_mem, mark.cu_mem, diff_order);
			fim_kernel_2<<<Blocks, Threads>>>(vel.frame, mark.cu_mem);
			fim_kernel_3<<<Blocks, Threads>>>(vel.frame, mark.cu_mem, endflag.cu_mem);
			cudaDeviceSynchronize();
			endflag.cu_stream_copy_d2h(racc_default_stream);
			cudaDeviceSynchronize();
			count++;
		} while (endflag[0] == true);
		endflag[0] == false;
		cudaDeviceSynchronize();
	}
}
#endif