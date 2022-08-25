#pragma once
#ifndef RACC_HOST_SEIS_FUN_BONES
#define RACC_HOST_SEIS_FUN_BONES
#include "racc_host_seis_0_struct.hpp"
namespace racc
{
	sp2vec ReadSRLine2D(std::string pathname);
	sp3vec ReadSRLine3D(std::string pathname);
	void CheckShotLine(Frame model, tp3vec shotline);
	void CheckReflectLine(Frame model, tp3vec reflectline);
	void _GetTimeAt(const dfield &time, tp3vec &position);
}
#endif