#pragma once
#ifndef RACC_DEVICE_FUN_MEAT
#define RACC_DEVICE_FUN_MEAT
#include "_cuda.h"
#include "racc_device_20_fun_bones.hpp"
namespace racc
{
  //***********************DEVICE_FUN************************
  inline __device__ bool CudaNextPermutation(int *p, int n)
  {
    int last = n - 1;
    int i, j, k;
    i = last;
    while (i > 0 && p[i] < p[i - 1])
      i--;
    if (i == 0)
      return false;
    k = i;
    for (j = last; j >= i; j--)
      if (p[j] > p[i - 1] && p[j] < p[k])
        k = j;
    SWAP(int, p[k], p[i - 1]);
    for (j = last, k = i; j > k; j--, k++)
      SWAP(int, p[j], p[k]);
    return true;
  }
  //
    template <typename eT>
  __device__ eT CudaMin(eT a, eT b)
  {
    if (a > b)
      return b;
    else
      return a;
  }
}
#endif