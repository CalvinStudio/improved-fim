#pragma once
#ifndef _TRAVEL_TIME_3D_FUNC_MEAT
#define _TRAVEL_TIME_3D_FUNC_MEAT
#include "travel_time_3d_3_func_bones.hpp"
namespace jarvis
{
	inline dcufld GetFresnel(fcufld &vel, const TimePoint3D &shot, const TimePoint3D &rece, int diff_order)
	{
		dcufld timea, timeb;
		tp3cuvec rece_tmp(MemType::pin, 1);
		travel_time_3d_module a;
		tp3cuvec shot_real(MemType::pin, 1);
		shot_real(0).x = shot.x;
		shot_real(0).y = shot.y;
		shot_real(0).z = shot.z;
		shot_real(0).time = 0;
		a.module_init(&vel, diff_order);
		a.cal_travel_time(shot_real, TravelType::normal);
		timea = a.GetTravelTimeField();
		rece_tmp(0).x = rece.x;
		rece_tmp(0).y = rece.y;
		rece_tmp(0).z = rece.z;
		a.get_time_at(rece_tmp);
		tp3cuvec rece_real(MemType::pin, 1);
		rece_real(0).x = rece.x;
		rece_real(0).y = rece.y;
		rece_real(0).z = rece.z;
		rece_real(0).time = 0;
		travel_time_3d_module b;
		b.module_init(&vel, diff_order);
		b.cal_travel_time(shot_real, TravelType::normal);
		timeb = b.GetTravelTimeField();
		dcufld fresnel(MemType::pin, vel.frame);
		field_for(vel.frame)
		{
			fresnel(i, j, k) = timea(i, j, k) + timeb(i, j, k) - rece_tmp(0).time;
		}
		return fresnel;
	}

	inline dcufld GetReflectFresnel(fcufld &vel, const TimePoint3D &shot, const TimePoint3D &rece, tp3cuvec &reflectline, int diff_order)
	{
#ifndef JARVIS_DEBUG
		check_recv_line_is_out_of_bounds(vel.frame, reflectline);
#endif
		dcufld timeS, timeSA, timeR, timeRA;
		tp3cuvec receS(MemType::pin, 1);
		travel_time_3d_module travelS;
		tp3cuvec shotS(MemType::pin, 1);
		shotS(0).x = shot.x;
		shotS(0).y = shot.y;
		shotS(0).z = shot.z;
		shotS(0).time = 0;
		travelS.module_init(&vel, diff_order);
		travelS.cal_travel_time(shotS, TravelType::normal);
		timeS = travelS.GetTravelTimeField();

		travelS.get_time_at(reflectline);
		travel_time_3d_module travelSA;
		travelSA.module_init(&vel, diff_order);
		travelSA.cal_travel_time(shotS, TravelType::normal);
		timeSA = travelSA.GetTravelTimeField();
		travel_time_3d_module travelR;
		tp3cuvec shotR(MemType::pin, 1);
		shotR(0).x = rece.x;
		shotR(0).y = rece.y;
		shotR(0).z = rece.z;
		shotR(0).time = 0;
		travelR.module_init(&vel, diff_order);
		travelR.cal_travel_time(shotR, TravelType::normal);
		timeR = travelR.GetTravelTimeField();

		travelR.get_time_at(reflectline);
		travel_time_3d_module travelRA;
		travelRA.module_init(&vel, diff_order);
		travelRA.cal_travel_time(reflectline, TravelType::normal);
		timeRA = travelRA.GetTravelTimeField();

		receS(0).x = shot.x;
		receS(0).y = shot.y;
		receS(0).z = shot.z;
		travelRA.get_time_at(receS);
		dcufld fresnel(MemType::pin, vel.frame);
		field_for(vel.frame)
		{
			fresnel(i, j, k) = min(timeSA(i, j, k) + timeR(i, j, k) - receS(0).time, timeS(i, j, k) + timeRA(i, j, k) - receS(0).time);
		}
		return fresnel;
	}
}
#endif