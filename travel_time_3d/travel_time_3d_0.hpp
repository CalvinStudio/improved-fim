#ifndef GPU_TRAVEL_TIME_3D_H
#define GPU_TRAVEL_TIME_3D_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "include/raccoon_01_device_bits/_racc_device_header_out.h"
#include "include/raccoon_10_host_seis_bits/_racc_host_seis_header_out.h"
namespace racc
{
	class travel_time_3d_module
	{
	public:
		int diff_order;
		tp3vec shotline;
		icufld mark;
		fcufld vel;
		dcufld time;
		//
		cuField<bool> endflag;
		//
		int Blocks;
		int Threads;
		//
		void init(const fcufld &_vel, int _diff_order);
		void cal_true_time();
		void cu_cal_time();
		void cal_refine_time(tp3vec &_shotline, int _extend_num, int _divide_num);
		void get_time_at(tp3vec &_position);
		void get_dev_time();
		dcufld GetTravelTimeField();
		void del_time();
		//
		void set_source(tp3vec &_shotline);
		void cuCalTime();
	};
}
#endif