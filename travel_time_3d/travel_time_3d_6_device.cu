#pragma once
#ifndef GPU_TRAVEL_TIME_3D_CU
#define GPU_TRAVEL_TIME_3D_CU
#include "travel_time_3d_5_meat.hpp"
namespace jarvis
{
	void travel_time_3d_module::cu_cal_time(tp3cuvec &_shotline)
	{
		set_source(_shotline);
		do
		{
			endflag[0] = true;
			endflag.cu_stream_copy_h2d(jarvis_default_stream);
			fim_kernel_0_cal_active_node<<<jarvis_cuda_kernel_size(vel_p->frame.n_elem)>>>(vel_p->frame, vel_p->cu_mem, time.cu_mem, mark.cu_mem, diff_order);
			fim_kernel_1_mark_new_active_node<<<jarvis_cuda_kernel_size(vel_p->frame.n_elem)>>>(vel_p->frame, vel_p->cu_mem, time.cu_mem, mark.cu_mem, diff_order);
			fim_kernel_2_set_as_converged_node<<<jarvis_cuda_kernel_size(vel_p->frame.n_elem)>>>(vel_p->frame, mark.cu_mem);
			fim_kernel_3_check_is_finishd<<<jarvis_cuda_kernel_size(vel_p->frame.n_elem)>>>(vel_p->frame, mark.cu_mem, endflag.cu_mem);
			endflag.cu_stream_copy_d2h(jarvis_default_stream);
			cudaStreamSynchronize(jarvis_default_stream);
		} while (endflag[0] == false);
		endflag[0] == true;
	}
}
#endif