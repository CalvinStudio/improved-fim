#pragma once
#ifndef RACC_DEVICE_FUN_BONES
#define RACC_DEVICE_FUN_BONES
#include "_cuda.h"
#include "racc_device_0_macro.hpp"
namespace racc
{
  //***********************DEVICE_FUN************************
  __device__ bool CudaNextPermutation(int *p, int n);
}
#endif