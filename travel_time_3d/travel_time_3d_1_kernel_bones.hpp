#ifndef TRAVEL_TIME_3D_KERNEL_HPP
#define TRAVEL_TIME_3D_KERNEL_HPP
#include "travel_time_3d_0.hpp"
namespace racc
{
    __device__ bool is_active(int tx, int ty, int tz, int idx, Frame RayVelGrid, int *_mark);

    __device__ double diff3d_1(int ix, int iy, int iz, Frame RayVelGrid, float *_vel, double *_time);

    __device__ double diff3d_1diag(int ix, int iy, int iz, Frame RayVelGrid, float *_vel, double *_time);

    __device__ double diff3d_2(int ix, int iy, int iz, Frame RayVelGrid, float *_vel, double *_time);

    __device__ double diff3d_2diag(int ix, int iy, int iz, Frame RayVelGrid, float *_vel, double *_time);

    __global__ void fim_kernel(Frame RayVelGrid, float *_vel, double *_time, int *_mark, int diff_order);

    __global__ void fim_kernel_0(Frame RayVelGrid, float *_vel, double *_time, int *_mark, int diff_order);

    __global__ void fim_kernel_1(Frame RayVelGrid, float *_vel, double *_time, int *_mark, int diff_order);

    __global__ void fim_kernel_2(Frame RayVelGrid, int *_mark);

    __global__ void fim_kernel_3(Frame RayVelGrid, int *_mark, bool *endflag);
}
#endif