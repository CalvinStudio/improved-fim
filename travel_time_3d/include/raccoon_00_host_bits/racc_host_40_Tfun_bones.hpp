#pragma once
#ifndef RACC_TFUN_BONES
#define RACC_TFUN_BONES
//**********************************Developer******************************************
// 2020.04.10 BY CAIWEI CALVIN CAI
//*************************************************************************************
#include "racc_host_31_Tmemory_meat.hpp"
//************************************ARRAY********************************************
namespace racc
{
	//两个数中的较小值
	template <typename eT>
	eT min(eT a, eT b);
	//三个数中的最小值
	template <typename eT>
	eT min(eT a, eT b, eT c);
	//两个数中的较大值
	template <typename eT>
	eT max(eT a, eT b);
	//三个数中的最大值
	template <typename eT>
	eT max(eT a, eT b, eT c);
	//一维数组中最小值
	template <typename eT>
	eT min(eT *a, int l); //一维数组中最小值
	//一维数组中最大值
	template <typename eT>
	eT max(eT *a, int l); //一维数组中最大值
	//二维数组中最小值
	template <typename eT>
	eT min(eT **a, uint32_t n_rows, uint32_t n_cols);
	//二维数组中最大值
	template <typename eT>
	eT max(eT **a, uint32_t n_rows, uint32_t n_cols);
	//三维数组中最小值
	template <typename eT>
	eT min(eT ***a, uint32_t n_rows, uint32_t n_cols, uint32_t n_slices);
	//三维数组中最大值
	template <typename eT>
	eT max(eT ***a, uint32_t n_rows, uint32_t n_cols, uint32_t n_slices);
	//一维数组中寻最小值和对应的点位置
	template <typename eT>
	void min(eT *a, uint64_t l, eT &min_val, uint64_t &pos);
	//一维数组中寻最大值和对应的点位置
	template <typename eT>
	void max(eT *a, uint64_t l, eT &max_val, uint64_t &pos);
	//二维数组中寻最小值和对应的点位置
	template <typename eT>
	void min(eT **a, int n_rows, uint32_t n_cols, eT *min_val, uint32_t &pos_x, uint32_t &pos_y);
	//二维数组中寻最大值和对应的点位置
	template <typename eT>
	void max(eT **a, uint32_t n_rows, uint32_t n_cols, eT *max_val, uint32_t &pos_x, uint32_t &pos_y);
	//查找1D数组绝对值的最小值
	template <typename eT>
	eT MinAbs(eT *a, uint64_t l);
	//查找1D数组绝对值的最大值
	template <typename eT>
	eT MaxAbs(eT *a, uint64_t l);
	//查找2D数组绝对值的最小值
	template <typename eT>
	eT MinAbs(eT **a, uint32_t n_rows, uint32_t n_cols);
	//查找2D数组绝对值的最大值
	template <typename eT>
	eT MaxAbs(eT **a, uint32_t n_rows, uint32_t n_cols);
	template <typename eT>
	void limit(eT &a, double lb, double rb);
	template <typename eT>
	void UnitVec(eT &a, eT &b);
	//*********************************************************************************
	//***********************Array transformation and operation************************
	// One dimensional array to three dimensional array.
	template <typename T>
	T ***TransArray1DTo3D(T *a_1, uint64_t l, uint32_t n_rows, uint32_t n_cols, uint32_t n_slices);
	// Three dimensional array to one dimensional array.
	template <typename T>
	T *TransArray3DTo1D(T ***a_3, uint32_t n_rows, uint32_t n_cols, uint32_t n_slices);
	//将数组从小到大排序,覆盖原数组
	template <typename T>
	void Sort(T *&a, uint64_t l);
	//计算ArrayIn的差分，输出ArrayOut
	template <typename T>
	void Diff(T *&a_i, uint64_t l, T di, T *a_o);
	template <typename T>
	void Swap(T &a, T &b);
	// Full Permutation
	template <typename T>
	bool NextPermutation(T *p, uint32_t n);
	// Show Permutation
	template <typename T>
	void ShowPermutation(T *p, uint32_t n);
	//*********************************************************************************
}
#endif