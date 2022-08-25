#pragma once
#ifndef RACC_HOST_TVEC_FUNCTION
#define RACC_HOST_TVEC_FUNCTION
//**********************************Developer*************************************
// 2020.04.10 BY CAIWEI CALVIN CAI
//********************************************************************************
#include "racc_host_71_Tvec_meat.hpp"
//********************************CLASS_TEMPLATE**********************************
namespace racc
{

	template <typename eT>
	inline Vec<eT> vec_range(eT a, double di, eT b)
	{
		long n_elem = int((b - a) / (1.0 * di)) + 1;
		Vec<eT> v(n_elem);
		for (int i = 0; i < n_elem; i++)
		{
			v(i) = a + di * i;
		}
		return v;
	}

	template <typename eT>
	inline Vec<eT> linspace(eT a, eT b, uint64_t n_elem)
	{
		Vec<eT> v(n_elem);
		eT di = (b - a) / (n_elem - 1);
		for (int i = 0; i < n_elem; i++)
		{
			v(i) = a + di * i;
		}
		return v;
	}

}

#endif
