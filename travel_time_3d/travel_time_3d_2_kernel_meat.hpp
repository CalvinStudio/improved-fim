#ifndef TRAVEL_TIME_3D_KERNEL_CU
#define TRAVEL_TIME_3D_KERNEL_CU
#include "travel_time_3d_1_kernel_bones.hpp"
namespace racc
{
    __global__ void fim_kernel(Frame RayVelGrid, float *_vel, double *_time, int *_mark, int diff_order)
    {
        SET_CUDA_FIELD_GRID_IDX(RayVelGrid, idx, tx, ty, tz);
        double trav = __DBL_MAX__;
        if (idx < RayVelGrid.n_elem)
        {
            // if (is_active(tx, ty, tz, RayVelGrid, _time) /*&& (_mark[idx] < 20)*/)
            // {
            if (diff_order == 11)
                trav = diff3d_1(tx, ty, tz, RayVelGrid, _vel, _time);

            if (diff_order == 12)
                trav = diff3d_1diag(tx, ty, tz, RayVelGrid, _vel, _time);

            if (diff_order == 21)
                trav = diff3d_2(tx, ty, tz, RayVelGrid, _vel, _time);

            if (diff_order == 22)
                trav = diff3d_2diag(tx, ty, tz, RayVelGrid, _vel, _time);

            if ((_time[idx] < __DBL_MAX__ / 2) &&
                (abs(_time[idx] - trav) < 1e-3))
                _mark[idx] = _mark[idx] + 1;

            _time[idx] = CudaMin(_time[idx], trav);
            // }
        }
    }

    //
    __global__ void fim_kernel_0(Frame RayVelGrid, float *_vel, double *_time, int *_mark, int diff_order)
    { //如果该点为激活点则进行旅行时计算
        SET_CUDA_FIELD_GRID_IDX(RayVelGrid, idx, tx, ty, tz);
        double old_trav = __DBL_MAX__;
        double trav = __DBL_MAX__;
        if (idx < RayVelGrid.n_elem)
        {
            if (_mark[idx] == 1)
            {
                old_trav = _time[idx];
                if (diff_order == 11)
                    trav = diff3d_1(tx, ty, tz, RayVelGrid, _vel, _time);
                if (diff_order == 12)
                    trav = diff3d_1diag(tx, ty, tz, RayVelGrid, _vel, _time);
                if (diff_order == 21)
                    trav = diff3d_2(tx, ty, tz, RayVelGrid, _vel, _time);
                if (diff_order == 22)
                    trav = diff3d_2diag(tx, ty, tz, RayVelGrid, _vel, _time);
                //
                _time[idx] = trav;
                //
                // if (abs(old_trav - trav) < 1e-4)
                {
                    _mark[idx] = 2; //标记为收敛点
                }
            }
        }
    }

    __global__ void fim_kernel_1(Frame RayVelGrid, float *_vel, double *_time, int *_mark, int diff_order)
    {
        //如果该点未激活且不是已收敛点，附近有待收敛点，则计算该点，后标记为激活点
        SET_CUDA_FIELD_GRID_IDX(RayVelGrid, idx, tx, ty, tz);
        double old_trav = __DBL_MAX__;
        double trav = __DBL_MAX__;
        if (idx < RayVelGrid.n_elem)
        {
            if (_mark[idx] != 2 && _mark[idx] != 3) //不是收敛点
            {
                if (is_active(tx, ty, tz, idx, RayVelGrid, _mark) && _mark[idx] != 1)
                {
                    old_trav = _time[idx];
                    if (diff_order == 11)
                        trav = diff3d_1(tx, ty, tz, RayVelGrid, _vel, _time);

                    if (diff_order == 12)
                        trav = diff3d_1diag(tx, ty, tz, RayVelGrid, _vel, _time);

                    if (diff_order == 21)
                        trav = diff3d_2(tx, ty, tz, RayVelGrid, _vel, _time);

                    if (diff_order == 22)
                        trav = diff3d_2diag(tx, ty, tz, RayVelGrid, _vel, _time);

                    if (old_trav > trav)
                    {
                        _time[idx] = trav;
                        _mark[idx] = 1; //标记为激活点
                    }
                }
            }
        }
    }

    __global__ void fim_kernel_4(Frame RayVelGrid, float *_vel, double *_time, int *_mark, int diff_order)
    {
        //如果该点未激活且不是已收敛点，附近有待收敛点，则计算该点，后标记为激活点
        SET_CUDA_FIELD_GRID_IDX(RayVelGrid, idx, tx, ty, tz);
        double old_trav = __DBL_MAX__;
        double trav = __DBL_MAX__;
        if (idx < RayVelGrid.n_elem)
        {
            if (_mark[idx] == 0) //如果是待收敛点
            {
                old_trav = _time[idx];
                if (diff_order == 11)
                    trav = diff3d_1(tx, ty, tz, RayVelGrid, _vel, _time);

                if (diff_order == 12)
                    trav = diff3d_1diag(tx, ty, tz, RayVelGrid, _vel, _time);

                if (diff_order == 21)
                    trav = diff3d_2(tx, ty, tz, RayVelGrid, _vel, _time);

                if (diff_order == 22)
                    trav = diff3d_2diag(tx, ty, tz, RayVelGrid, _vel, _time);

                if (old_trav > trav)
                {
                    _time[idx] = trav;
                }
                if (abs(old_trav - trav) < 1e-6)
                {
                    _mark[idx] = 3;
                }
            }
        }
    }

    __global__ void fim_kernel_2(Frame RayVelGrid, int *_mark)
    { //将待收敛点标记为0;
        SET_CUDA_FIELD_GRID_IDX(RayVelGrid, idx, tx, ty, tz);
        if (idx < RayVelGrid.n_elem)
        {
            if (_mark[idx] == 2)
            {
                _mark[idx] = 100;
            }
        }
    }

    __global__ void fim_kernel_3(Frame RayVelGrid, int *_mark, bool *endflag)
    {
        SET_CUDA_FIELD_GRID_IDX(RayVelGrid, idx, tx, ty, tz);
        if (idx < RayVelGrid.n_elem)
        {
            if (_mark[idx] == 1)
            {
                endflag[0] = true;
            }
        }
    }

    __device__ bool is_active(int tx, int ty, int tz, int idx, Frame RayVelGrid, int *_mark)
    { //! _mark=-1:未激活的点;    1:激活点L;   2:收敛点
        //!周围如果有待收敛的点 则激活该点
        SET_MODEL_GRID_N(RayVelGrid);
        int lax = -1;
        int rax = 1;
        int lay = -1;
        int ray = 1;
        int laz = -1;
        int raz = 1;
        if (tx <= 0)
            lax = 1;
        if (tx >= n_rows - 1)
            rax = -1;
        if (ty <= 0)
            lay = 1;
        if (ty >= n_cols - 1)
            ray = -1;
        if (tz <= 0)
            laz = 1;
        if (tz >= n_slices - 1)
            raz = -1;
        if ((_mark[idx + lax] == 2) || (_mark[idx + rax] == 2) ||
            (_mark[idx + lay * n_rows] == 2) || (_mark[idx + ray * n_rows] == 2) ||
            (_mark[idx + laz * n_elem_slice] == 2) || (_mark[idx + raz * n_elem_slice] == 2))
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    __device__ double diff3d_1(int ix, int iy, int iz, Frame RayVelGrid, float *_vel, double *_time)
    {
        SET_MODEL_GRID_N(RayVelGrid);
        double d_rows = RayVelGrid.d_rows;
        double a, b, c, slown;
        double trav, travm;
        int xtmp, ytmp, ztmp;
        double ttnminx, ttnminy, ttnminz;
        int ax, ay, az;
        trav = __DBL_MAX__, travm = __DBL_MAX__;
        xtmp = ix;
        ytmp = iy;
        ztmp = iz;
        if (ix == n_rows)
            xtmp = n_rows - 1;
        if (iy == n_cols)
            ytmp = n_cols - 1;
        if (iz == n_slices)
            ztmp = n_slices - 1;
        slown = 1.0f / F_(_vel, xtmp, ytmp, ztmp);

        if ((ix >= 0 && ix < n_rows) && (iy >= 0 && iy < n_cols) && (iz >= 0 && iz < n_slices))
        {
            if (ix - 1 < 0)
                ax = 1;
            else if (ix + 1 >= n_rows)
                ax = -1;
            else if (F_(_time, (ix + 1), iy, iz) <= F_(_time, (ix - 1), iy, iz))
                ax = 1;
            else
                ax = -1;

            if (iy - 1 < 0)
                ay = 1;
            else if (iy + 1 >= n_cols)
                ay = -1;
            else if (F_(_time, ix, (iy + 1), iz) <= F_(_time, ix, (iy - 1), iz))
                ay = 1;
            else
                ay = -1;

            if (iz - 1 < 0)
                az = 1;
            else if (iz + 1 >= n_slices)
                az = -1;
            else if (F_(_time, ix, iy, (iz + 1)) <= F_(_time, ix, iy, (iz - 1)))
                az = 1;
            else
                az = -1;

            ttnminx = F_(_time, (ix + ax), iy, iz);
            ttnminy = F_(_time, ix, (iy + ay), iz);
            ttnminz = F_(_time, ix, iy, (iz + az));

            int p[3] = {0, 1, 2}; //用于123全排列
            double ttnmin[3] = {ttnminx, ttnminy, ttnminz};
            do
            {
                if (ttnmin[p[0]] <= ttnmin[p[1]] && ttnmin[p[1]] <= ttnmin[p[2]])
                {
                    c = ttnmin[p[0]];
                    b = ttnmin[p[1]];
                    a = ttnmin[p[2]];
                }
            } while (CudaNextPermutation(p, 3));
            trav = c + d_rows * slown;
            if (trav > b)
            {
                trav = (b + c +
                        sqrt(-b * b - c * c + 2.0f * b * c +
                             2.0f * (d_rows * d_rows * slown * slown))) /
                       2.0f;
                if (trav > a)
                {
                    trav =
                        (2.0f * (a + b + c) + sqrt(4.0f * (a + b + c) * (a + b + c) -
                                                   12.0f * (a * a + b * b + c * c -
                                                            (d_rows * d_rows * slown * slown)))) /
                        6.0f;
                }
            }
            travm = trav;
        }
        return travm;
    }

    __device__ double diff3d_1diag(int ix, int iy, int iz, Frame RayVelGrid, float *_vel, double *_time)
    {
        // SET_MODEL_GRID_N(RayVelGrid);
        // double d_rows = RayVelGrid.d_rows;
        // double a, b, c, slown;
        // double trav, travm;
        // int xtmp, ytmp, ztmp;
        // double ttnminx, ttnminy, ttnminz;
        // int ax, ay, az;
        // trav = __DBL_MAX__, travm = __DBL_MAX__;
        // xtmp = ix;
        // ytmp = iy;
        // ztmp = iz;
        // if (ix == n_rows)
        //     xtmp = n_rows - 1;
        // if (iy == n_cols)
        //     ytmp = n_cols - 1;
        // if (iz == n_slices)
        //     ztmp = n_slices - 1;
        // slown = 1.0f / F_(_vel, xtmp, ytmp, ztmp);

        // if ((ix >= 0 && ix < n_rows) && (iy >= 0 && iy < n_cols) && (iz >= 0 && iz < n_slices))
        // {
        //     if (ix - 1 < 0)
        //         ax = 1;
        //     else if (ix + 1 >= n_rows)
        //         ax = -1;
        //     else if (F_(_time, (ix + 1), iy, iz) <= F_(_time, (ix - 1), iy, iz))
        //         ax = 1;
        //     else
        //         ax = -1;

        //     if (iy - 1 < 0)
        //         ay = 1;
        //     else if (iy + 1 >= n_cols)
        //         ay = -1;
        //     else if (F_(_time, ix, (iy + 1), iz) <= F_(_time, ix, (iy - 1), iz))
        //         ay = 1;
        //     else
        //         ay = -1;

        //     if (iz - 1 < 0)
        //         az = 1;
        //     else if (iz + 1 >= n_slices)
        //         az = -1;
        //     else if (F_(_time, ix, iy, (iz + 1)) <= F_(_time, ix, iy, (iz - 1)))
        //         az = 1;
        //     else
        //         az = -1;

        //     ttnminx = F_(_time, (ix + ax), iy, iz);
        //     ttnminy = F_(_time, ix, (iy + ay), iz);
        //     ttnminz = F_(_time, ix, iy, (iz + az));

        //     int p[3] = {0, 1, 2}; //用于123全排列
        //     double ttnmin[3] = {ttnminx, ttnminy, ttnminz};
        //     do
        //     {
        //         if (ttnmin[p[0]] <= ttnmin[p[1]] && ttnmin[p[1]] <= ttnmin[p[2]])
        //         {
        //             c = ttnmin[p[0]];
        //             b = ttnmin[p[1]];
        //             a = ttnmin[p[2]];
        //         }
        //     } while (CudaNextPermutation(p, 3));
        //     trav = c + d_rows * slown;
        //     if (trav > b)
        //     {
        //         trav = (b + c +
        //                 sqrt(-b * b - c * c + 2.0f * b * c +
        //                      2.0f * (d_rows * d_rows * slown * slown))) /
        //                2.0f;
        //         if (trav > a)
        //         {
        //             trav =
        //                 (2.0f * (a + b + c) + sqrt(4.0f * (a + b + c) * (a + b + c) -
        //                                           12.0f * (a * a + b * b + c * c -
        //                                                   (d_rows * d_rows * slown * slown)))) /
        //                 6.0f;
        //         }
        //     }
        //     travm = trav;
        // }
        // return travm;

        SET_MODEL_GRID_N(RayVelGrid);
        double d_rows = RayVelGrid.d_rows;
        double a, b, c;
        double slown;
        double trav, travm;
        int xtmp, ytmp, ztmp;
        double ttnminx, ttnminy, ttnminz;
        int ax, ay, az;
        double trav_diag;
        trav = __DBL_MAX__, travm = __DBL_MAX__, trav_diag = __DBL_MAX__;

        xtmp = ix;
        ytmp = iy;
        ztmp = iz;
        if (ix == n_rows)
            xtmp = n_rows - 1;
        if (iy == n_cols)
            ytmp = n_cols - 1;
        if (iz == n_slices)
            ztmp = n_slices - 1;

        slown = 1.0f / F_(_vel, xtmp, ytmp, ztmp);

        if ((ix >= 0 && ix < n_rows) && (iy >= 0 && iy < n_cols) && (iz >= 0 && iz < n_slices))
        {
            if (ix - 1 < 0)
                ax = 1;
            else if (ix + 1 >= n_rows)
                ax = -1;
            else if (F_(_time, (ix + 1), iy, iz) <= F_(_time, (ix - 1), iy, iz))
                ax = 1;
            else
                ax = -1;

            if (iy - 1 < 0)
                ay = 1;
            else if (iy + 1 >= n_cols)
                ay = -1;
            else if (F_(_time, ix, (iy + 1), iz) <= F_(_time, ix, (iy - 1), iz))
                ay = 1;
            else
                ay = -1;

            if (iz - 1 < 0)
                az = 1;
            else if (iz + 1 >= n_slices)
                az = -1;
            else if (F_(_time, ix, iy, (iz + 1)) <= F_(_time, ix, iy, (iz - 1)))
                az = 1;
            else
                az = -1;

            ttnminx = F_(_time, (ix + ax), iy, iz);
            ttnminy = F_(_time, ix, (iy + ay), iz);
            ttnminz = F_(_time, ix, iy, (iz + az));

            int p[3] = {0, 1, 2}; //用于123全排列
            double ttnmin[3] = {ttnminx, ttnminy, ttnminz};
            do
            {
                if (ttnmin[p[0]] <= ttnmin[p[1]] && ttnmin[p[1]] <= ttnmin[p[2]])
                {
                    c = ttnmin[p[0]];
                    b = ttnmin[p[1]];
                    a = ttnmin[p[2]];
                }
            } while (CudaNextPermutation(p, 3));

            trav = c + d_rows * slown;

            if (trav > b)
            {
                trav = (b + c +
                        sqrt(-b * b - c * c + 2.0f * b * c +
                             2.0f * (d_rows * d_rows * slown * slown))) /
                       2.0f;

                if (trav > a)
                {
                    trav =
                        (2.0f * (a + b + c) + sqrt(4.0f * (a + b + c) * (a + b + c) -
                                                   12.0f * (a * a + b * b + c * c -
                                                            (d_rows * d_rows * slown * slown)))) /
                        6.0f;

                    trav_diag =
                        F_(_time, (ix + ax), (iy + ay), (iz + az)) + sqrt(3.0f) * d_rows * slown;

                    if (trav_diag < trav)
                    {
                        trav = trav_diag;
                    }
                }
            }
            travm = trav;
        }
        return travm;
    }

    __device__ double diff3d_2(int ix, int iy, int iz, Frame RayVelGrid,
                               float *_vel, double *_time)
    {
        SET_MODEL_GRID_N(RayVelGrid);
        double d_rows = RayVelGrid.d_rows;
        double d_cols = RayVelGrid.d_cols;
        double d_slices = RayVelGrid.d_slices;
        double a, b, c, slown;
        double a2, b2, c2;
        double aa, bb, cc;
        double aa1, bb1, cc1;
        double aa2, bb2, cc2;
        double aa3, bb3, cc3;
        double trav, trav1, trav2, trav3, trav4, trav5, trav6, travm, rd;
        int xtmp, ytmp, ztmp;
        double ttnminx, ttnminy, ttnminz;
        double ttnminx2, ttnminy2, ttnminz2;
        ttnminx2 = __DBL_MAX__, ttnminy2 = __DBL_MAX__, ttnminz2 = __DBL_MAX__;
        int ax, ay, az;

        trav = 0.0f, trav1 = __DBL_MAX__, trav2 = __DBL_MAX__, trav3 = __DBL_MAX__,
        trav4 = __DBL_MAX__, trav5 = __DBL_MAX__, trav6 = __DBL_MAX__, travm = __DBL_MAX__, rd = 0.0f;
        xtmp = ix;
        ytmp = iy;
        ztmp = iz;
        if (ix == n_rows)
            xtmp = n_rows - 1;
        if (iy == n_cols)
            ytmp = n_cols - 1;
        if (iz == n_slices)
            ztmp = n_slices - 1;
        slown = 1.0f / F_(_vel, xtmp, ytmp, ztmp);
        if ((ix >= 0 && ix < n_rows) && (iy >= 0 && iy < n_cols) && (iz >= 0 && iz < n_slices))
        {
            if (ix - 1 < 0)
                ax = 1;
            else if (ix + 1 >= n_rows)
                ax = -1;
            else if (F_(_time, (ix + 1), iy, iz) < F_(_time, (ix - 1), iy, iz))
                ax = 1;
            else
                ax = -1;

            if (iy - 1 < 0)
                ay = 1;
            else if (iy + 1 >= n_cols)
                ay = -1;
            else if (F_(_time, ix, (iy + 1), iz) < F_(_time, ix, (iy - 1), iz))
                ay = 1;
            else
                ay = -1;

            if (iz - 1 < 0)
                az = 1;
            else if (iz + 1 >= n_slices)
                az = -1;
            else if (F_(_time, ix, iy, (iz + 1)) < F_(_time, ix, iy, (iz - 1)))
                az = 1;
            else
                az = -1;

            ttnminx = F_(_time, (ix + ax), iy, iz);
            ttnminy = F_(_time, ix, (iy + ay), iz);
            ttnminz = F_(_time, ix, iy, (iz + az));

            ttnminx2 = F_(_time, (ix + 2 * ax), iy, iz);
            ttnminy2 = F_(_time, ix, (iy + 2 * ay), iz);
            ttnminz2 = F_(_time, ix, iy, (iz + 2 * az));

            int p[3] = {0, 1, 2}; //用于123全排列
            double ttnmin[3] = {ttnminx, ttnminy, ttnminz};
            double ttnmin2[3] = {ttnminx2, ttnminy2, ttnminz2};
            do
            {
                if (ttnmin[p[0]] <= ttnmin[p[1]] && ttnmin[p[1]] <= ttnmin[p[2]])
                {
                    c = ttnmin[p[0]];
                    b = ttnmin[p[1]];
                    a = ttnmin[p[2]];
                    c2 = ttnmin2[p[0]];
                    b2 = ttnmin2[p[1]];
                    a2 = ttnmin2[p[2]];
                }
            } while (CudaNextPermutation(p, 3));

            trav = c + d_rows * slown;

            if (trav > b)
            {
                trav1 = (b + c +
                         sqrt(-b * b - c * c + 2.0f * b * c +
                              2.0f * d_rows * d_rows * slown * slown)) /
                        2.0f;
                if (trav1 < trav)
                {
                    trav = trav1;
                }
                if (c2 < c)
                {
                    trav2 =
                        (3.0f * (4.0f * c - c2) + 4.0f * b +
                         2.0f * sqrt(13 * slown * slown * d_rows * d_rows -
                                     (4.0f * c - c2 - 3.0f * b) * (4.0f * c - c2 - 3.0f * b))) /
                        13.0f;
                    if (trav2 < trav)
                    {
                        trav = trav2;
                    }
                    if (b2 < b)
                    {
                        trav3 = ((4.0f * c - c2) + (4.0f * b - b2) +
                                 sqrt(8.0f * (slown * slown * d_rows * d_rows) -
                                      ((4.0f * c - c2) - (4.0f * b - b2)) *
                                          ((4.0f * c - c2) - (4.0f * b - b2)))) /
                                6.0f;
                    }
                    if (trav3 < trav)
                    {
                        trav = trav3;
                    }
                }
                if (trav > a)
                {
                    trav =
                        (2.0f * (a + b + c) + sqrt(4.0f * (a + b + c) * (a + b + c) -
                                                   12.0f * (a * a + b * b + c * c -
                                                            d_rows * d_rows * (slown * slown)))) /
                        6.0f;

                    if (c2 < c)
                    {
                        aa1 = 9.0f / (4.0f * d_rows * d_rows);
                        bb1 = (-24.0f * c + 6.0f * c2) / (4.0f * d_rows * d_rows);
                        cc1 = ((4.0f * c - c2) * (4.0f * c - c2)) / (4.0f * d_rows * d_rows);

                        aa2 = 1.0f / (d_cols * d_cols);
                        bb2 = -2.0f * b / (d_cols * d_cols);
                        cc2 = b * b / (d_cols * d_cols);

                        aa3 = 1.0f / (d_slices * d_slices);
                        bb3 = -2.0f * a / (d_slices * d_slices);
                        cc3 = a * a / (d_slices * d_slices);

                        aa = aa1 + aa2 + aa3;
                        bb = bb1 + bb2 + bb3;
                        cc = cc1 + cc2 + cc3 - (slown * slown);

                        rd = bb * bb - 4.0f * aa * cc;
                        if (rd < 0.0f)
                            rd = 0.0f;
                        trav4 = (-bb + sqrt(rd)) / (2.0f * aa);
                        if (trav4 < trav)
                        {
                            trav = trav4;
                        }

                        if (b2 < b)
                        {
                            aa1 = 9.0f / (4.0f * d_rows * d_rows);
                            bb1 = (-24.0f * c + 6.0f * c2) / (4.0f * d_rows * d_rows);
                            cc1 = ((4.0f * c - c2) * (4.0f * c - c2)) / (4.0f * d_rows * d_rows);
                            aa2 = 9.0f / (4.0f * d_cols * d_cols);
                            bb2 = (-24.0f * b + 6.0f * b2) / (4.0f * d_cols * d_cols);
                            cc2 = ((4.0f * b - b2) * (4.0f * b - b2)) / (4.0f * d_cols * d_cols);

                            aa3 = 1.0f / (d_slices * d_slices);
                            bb3 = -2.0f * a / (d_slices * d_slices);
                            cc3 = a * a / (d_slices * d_slices);

                            aa = aa1 + aa2 + aa3;
                            bb = bb1 + bb2 + bb3;
                            cc = cc1 + cc2 + cc3 - (slown * slown);
                            rd = bb * bb - 4.0f * aa * cc;
                            if (rd < 0.0f)
                                rd = 0.0f;
                            trav5 = (-bb + sqrt(rd)) / (2.0f * aa);
                            if (trav5 < trav)
                            {
                                trav = trav5;
                            }
                            if (a2 < a)
                            {
                                aa1 = 9.0f / (4.0f * d_rows * d_rows);
                                bb1 = (-24.0f * c + 6.0f * c2) / (4.0f * d_rows * d_rows);
                                cc1 = ((4.0f * c - c2) * (4.0f * c - c2)) / (4.0f * d_rows * d_rows);
                                //
                                aa2 = 9.0f / (4.0f * d_cols * d_cols);
                                bb2 = (-24.0f * b + 6.0f * b2) / (4.0f * d_cols * d_cols);
                                cc2 = ((4.0f * b - b2) * (4.0f * b - b2)) / (4.0f * d_cols * d_cols);
                                //
                                aa3 = 9.0f / (4.0f * d_slices * d_slices);
                                bb3 = (-24.0f * a + 6.0f * a2) / (4.0f * d_slices * d_slices);
                                cc3 = ((4.0f * a - a2) * (4.0f * a - a2)) / (4.0f * d_slices * d_slices);
                                //
                                aa = aa1 + aa2 + aa3;
                                bb = bb1 + bb2 + bb3;
                                cc = cc1 + cc2 + cc3 - (slown * slown);
                                rd = bb * bb - 4.0f * aa * cc;
                                if (rd < 0.0f)
                                    rd = 0.0f;
                                trav6 = (-bb + sqrt(rd)) / (2.0f * aa);
                                if (trav6 < trav)
                                {
                                    trav = trav6;
                                }
                            }
                        }
                    }
                }
            }
            travm = trav;
        }
        return travm;
    }

    // __device__ double diff3d_2(int ix, int iy, int iz, Frame RayVelGrid,
    //                            float *_vel, double *ttn)
    // {
    //     SET_MODEL_GRID_ND(RayVelGrid);
    //     int nnx = n_rows;
    //     int nny = n_cols;
    //     int nnz = n_slices;
    //     float dnx = d_rows;
    //     float dny = d_cols;
    //     float dnz = d_slices;
    //     float a, b, c, slown;
    //     float a2, b2, c2;
    //     float aa, bb, cc;
    //     float aa1, bb1, cc1;
    //     float aa2, bb2, cc2;
    //     float aa3, bb3, cc3;
    //     float trav, trav11, trav2, trav3, trav4, travm, rd;
    //     int xtmp, ytmp, ztmp;
    //     float ttnminx, ttnminy, ttnminz;
    //     float ttnminx2, ttnminy2, ttnminz2;
    //     ttnminx2 = __DBL_MAX__, ttnminy2 = __DBL_MAX__, ttnminz2 = __DBL_MAX__;
    //     int ax, ay, az;
    //     int axx = 0, ayy = 0, azz = 0;

    //     trav = __DBL_MAX__, trav11 = __DBL_MAX__, trav2 = __DBL_MAX__, trav3 = __DBL_MAX__,
    //     trav4 = __DBL_MAX__, travm = __DBL_MAX__, rd = 0.0f;
    //     xtmp = ix;
    //     ytmp = iy;
    //     ztmp = iz;
    //     if (ix == nnx)
    //         xtmp = nnx - 1;
    //     if (iy == nny)
    //         ytmp = nny - 1;
    //     if (iz == nnz)
    //         ztmp = nnz - 1;
    //     slown = 1.0f / _vel[xtmp * n_slices + ytmp * n_rows * n_slices + ztmp];
    //     if ((ix >= 0 && ix < nnx) && (iy >= 0 && iy < nny) && (iz >= 0 && iz < nnz))
    //     {
    //         if (ix - 2 < 0)
    //         {
    //             axx = 1;
    //         }
    //         if (ix + 2 >= nnx)
    //         {
    //             axx = -1;
    //         }

    //         if (iy - 2 < 0)
    //         {
    //             ayy = 1;
    //         }
    //         if (iy + 2 >= nny)
    //         {
    //             ayy = -1;
    //         }

    //         if (iz - 2 < 0)
    //         {
    //             azz = 1;
    //         }
    //         if (iz + 2 >= nnz)
    //         {
    //             azz = -1;
    //         }

    //         if (ix - 1 < 0)
    //         {
    //             ax = 1;
    //         }
    //         else if (ix + 1 >= nnx)
    //         {
    //             ax = -1;
    //         }
    //         else if (ttn[(ix + 1) * nnz + iy * nnx * nnz + iz] <=
    //                  ttn[(ix - 1) * nnz + iy * nnx * nnz + iz])
    //         {
    //             ax = 1;
    //         }
    //         else
    //         {
    //             ax = -1;
    //         }

    //         if (iy - 1 < 0)
    //         {
    //             ay = 1;
    //         }
    //         else if (iy + 1 >= nny)
    //         {
    //             ay = -1;
    //         }
    //         else if (ttn[ix * nnz + (iy + 1) * nnx * nnz + iz] <=
    //                  ttn[ix * nnz + (iy - 1) * nnx * nnz + iz])
    //         {
    //             ay = 1;
    //         }
    //         else
    //         {
    //             ay = -1;
    //         }

    //         if (iz - 1 < 0)
    //         {
    //             az = 1;
    //         }
    //         else if (iz + 1 >= nnz)
    //         {
    //             az = -1;
    //         }
    //         else if (ttn[ix * nnz + iy * nnx * nnz + iz + 1] <=
    //                  ttn[ix * nnz + iy * nnx * nnz + iz - 1])
    //         {
    //             az = 1;
    //         }
    //         else
    //         {
    //             az = -1;
    //         }

    //         ttnminx = ttn[(ix + ax) * nnz + iy * nnz * nnx + iz];
    //         ttnminy = ttn[ix * nnz + (iy + ay) * nnz * nnx + iz];
    //         ttnminz = ttn[ix * nnz + iy * nnz * nnx + iz + az];

    //         if (axx == 0)
    //             ttnminx2 = ttn[(ix + 2 * ax) * nnz + iy * nnz * nnx + iz];

    //         if (ayy == 0)
    //             ttnminy2 = ttn[ix * nnz + (iy + 2 * ay) * nnz * nnx + iz];

    //         if (azz == 0)
    //             ttnminz2 = ttn[ix * nnz + iy * nnz * nnx + iz + 2 * az];

    //         if (ttnminx <= ttnminy && ttnminy <= ttnminz)
    //         {
    //             c = ttnminx;
    //             b = ttnminy;
    //             a = ttnminz;
    //             c2 = ttnminx2;
    //             b2 = ttnminy2;
    //             a2 = ttnminz2;
    //         }
    //         else if (ttnminx <= ttnminz && ttnminz <= ttnminy)
    //         {
    //             c = ttnminx;
    //             b = ttnminz;
    //             a = ttnminy;
    //             c2 = ttnminx2;
    //             b2 = ttnminz2;
    //             a2 = ttnminy2;
    //         }
    //         else if (ttnminy <= ttnminz && ttnminz <= ttnminx)
    //         {
    //             c = ttnminy;
    //             b = ttnminz;
    //             a = ttnminx;
    //             c2 = ttnminy2;
    //             b2 = ttnminz2;
    //             a2 = ttnminx2;
    //         }
    //         else if (ttnminy <= ttnminx && ttnminx <= ttnminz)
    //         {
    //             c = ttnminy;
    //             b = ttnminx;
    //             a = ttnminz;
    //             c2 = ttnminy2;
    //             b2 = ttnminx2;
    //             a2 = ttnminz2;
    //         }
    //         else if (ttnminz <= ttnminx && ttnminx <= ttnminy)
    //         {
    //             c = ttnminz;
    //             b = ttnminx;
    //             a = ttnminy;
    //             c2 = ttnminz2;
    //             b2 = ttnminx2;
    //             a2 = ttnminy2;
    //         }
    //         else if (ttnminz <= ttnminy && ttnminy <= ttnminx)
    //         {
    //             c = ttnminz;
    //             b = ttnminy;
    //             a = ttnminx;
    //             c2 = ttnminz2;
    //             b2 = ttnminy2;
    //             a2 = ttnminx2;
    //         }

    //         trav = c + dnx * slown;

    //         if (trav > b)
    //         {
    //             trav11 = (b + c +
    //                       sqrtf(-b * b - c * c + 2.0f * b * c +
    //                             2.0f * dnx * dnx * (slown * slown))) /
    //                      2.0f;

    //             if (trav11 < trav)
    //             {
    //                 trav = trav11;
    //             }

    //             // trav_diag = t_diag2 + sqrtf(2.0f)*dnx*slown;
    //             // if (trav_diag < trav)
    //             //{
    //             //	trav = trav_diag;
    //             //}

    //             // if (c2 > c)
    //             //{
    //             //	trav_diag = t_diag2 + sqrtf(2.0f)*dnx*slown;

    //             //	if (trav_diag < trav)
    //             //	{
    //             //		trav = trav_diag;
    //             //	}
    //             //}

    //             if (c2 < c)
    //             {
    //                 trav2 =
    //                     (3.0f * (4.0f * c - c2) + 4.0f * b +
    //                      2.0f * sqrtf(13 * slown * slown * dnx * dnx -
    //                                  (4.0f * c - c2 - 3.0f * b) * (4.0f * c - c2 - 3.0f * b))) /
    //                     13.0f;
    //                 if (trav2 < trav)
    //                 {
    //                     trav = trav2;
    //                 }
    //                 if (b2 < b)
    //                 {
    //                     trav3 = ((4.0f * c - c2) + (4.0f * b - b2) +
    //                              sqrtf(8.0f * (slown * slown * dnx * dnx) -
    //                                    ((4.0f * c - c2) - (4.0f * b - b2)) *
    //                                        ((4.0f * c - c2) - (4.0f * b - b2)))) /
    //                             6.0f;
    //                 }
    //                 if (trav3 < trav)
    //                 {
    //                     trav = trav3;
    //                 }
    //             }

    //             if (trav > a)
    //             {
    //                 trav = (2.0f * (a + b + c) +
    //                         sqrtf(4.0f * (a + b + c) * (a + b + c) -
    //                               12.0f * (a * a + b * b + c * c -
    //                                       1.0f * dnx * dnx * (slown * slown)))) /
    //                        6.0f;

    //                 // if (c2 >= c)
    //                 //{
    //                 //	trav_diag = ttn[(ix + ax)*nnz + (iy + ay) * nnz*nnx + iz + az] +
    //                 // sqrt(3.0f)*dnx*slown;

    //                 //	if (trav_diag < trav)
    //                 //	{
    //                 //		trav = trav_diag;
    //                 //	}
    //                 //}

    //                 if (c2 < c)
    //                 {
    //                     aa1 = 9.0f / (4.0f * dnx * dnx);
    //                     bb1 = (-24.0f * c + 6.0f * c2) / (4.0f * dnx * dnx);
    //                     cc1 = ((4.0f * c - c2) * (4.0f * c - c2)) / (4.0f * dnx * dnx);

    //                     aa2 = 1.0f / (dny * dny);
    //                     bb2 = -2.0f * b / (dny * dny);
    //                     cc2 = b * b / (dny * dny);

    //                     aa3 = 1.0f / (dnz * dnz);
    //                     bb3 = -2.0f * a / (dnz * dnz);
    //                     cc3 = a * a / (dnz * dnz);

    //                     aa = aa1 + aa2 + aa3;
    //                     bb = bb1 + bb2 + bb3;
    //                     cc = cc1 + cc2 + cc3 - (slown * slown);

    //                     rd = bb * bb - 4.0f * aa * cc;
    //                     if (rd < 0.0f)
    //                         rd = 0.0f;
    //                     trav2 = (-bb + sqrt(rd)) / (2.0f * aa);

    //                     if (trav2 < trav)
    //                     {
    //                         trav = trav2;
    //                     }

    //                     if (b2 < b)
    //                     {
    //                         aa1 = 9.0f / (4.0f * dnx * dnx);
    //                         bb1 = (-24.0f * c + 6.0f * c2) / (4.0f * dnx * dnx);
    //                         cc1 = ((4.0f * c - c2) * (4.0f * c - c2)) / (4.0f * dnx * dnx);

    //                         aa2 = 9.0f / (4.0f * dny * dny);
    //                         bb2 = (-24.0f * b + 6.0f * b2) / (4.0f * dny * dny);
    //                         cc2 = ((4.0f * b - b2) * (4.0f * b - b2)) / (4.0f * dny * dny);

    //                         aa3 = 1.0f / (dnz * dnz);
    //                         bb3 = -2.0f * a / (dnz * dnz);
    //                         cc3 = a * a / (dnz * dnz);

    //                         aa = aa1 + aa2 + aa3;
    //                         bb = bb1 + bb2 + bb3;
    //                         cc = cc1 + cc2 + cc3 - (slown * slown);

    //                         rd = bb * bb - 4.0f * aa * cc;
    //                         if (rd < 0.0f)
    //                             rd = 0.0f;
    //                         trav3 = (-bb + sqrt(rd)) / (2.0f * aa);

    //                         if (trav3 < trav)
    //                         {
    //                             trav = trav3;
    //                         }

    //                         if (a2 < a)
    //                         {
    //                             aa1 = 9.0f / (4.0f * dnx * dnx);
    //                             bb1 = (-24.0f * c + 6.0f * c2) / (4.0f * dnx * dnx);
    //                             cc1 = ((4.0f * c - c2) * (4.0f * c - c2)) / (4.0f * dnx * dnx);

    //                             aa2 = 9.0f / (4.0f * dny * dny);
    //                             bb2 = (-24.0f * b + 6.0f * b2) / (4.0f * dny * dny);
    //                             cc2 = ((4.0f * b - b2) * (4.0f * b - b2)) / (4.0f * dny * dny);

    //                             aa3 = 9.0f / (4.0f * dnz * dnz);
    //                             bb3 = (-24.0f * a + 6.0f * a2) / (4.0f * dnz * dnz);
    //                             cc3 = ((4.0f * a - a2) * (4.0f * a - a2)) / (4.0f * dnz * dnz);

    //                             aa = aa1 + aa2 + aa3;
    //                             bb = bb1 + bb2 + bb3;
    //                             cc = cc1 + cc2 + cc3 - (slown * slown);

    //                             rd = bb * bb - 4.0f * aa * cc;
    //                             if (rd < 0.0f)
    //                                 rd = 0.0f;
    //                             trav4 = (-bb + sqrt(rd)) / (2.0f * aa);

    //                             if (trav4 < trav)
    //                             {
    //                                 trav = trav4;
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //         travm = trav;
    //     }
    //     return travm;
    // }

    // __device__ double diff3d_2diag(int ix, int iy, int iz, Frame RayVelGrid,
    //                                float *_vel, double *ttn)
    // {
    //     SET_MODEL_GRID_ND(RayVelGrid);
    //     int nnx = n_rows;
    //     int nny = n_cols;
    //     int nnz = n_slices;
    //     float dnx = d_rows;
    //     float dny = d_cols;
    //     float dnz = d_slices;
    //     float a, b, c, slown;
    //     float a2, b2, c2;
    //     float aa, bb, cc;
    //     float aa1, bb1, cc1;
    //     float aa2, bb2, cc2;
    //     float aa3, bb3, cc3;
    //     float trav, trav11, trav2, trav3, trav4, travm, rd;
    //     int xtmp, ytmp, ztmp;
    //     float ttnminx, ttnminy, ttnminz;
    //     float ttnminx2, ttnminy2, ttnminz2;
    //     ttnminx2 = __DBL_MAX__, ttnminy2 = __DBL_MAX__, ttnminz2 = __DBL_MAX__;
    //     int ax, ay, az;
    //     int axx = 0, ayy = 0, azz = 0;

    //     float trav_diag;
    //     trav = __DBL_MAX__, trav11 = __DBL_MAX__, trav2 = __DBL_MAX__, trav3 = __DBL_MAX__,
    //     trav4 = __DBL_MAX__, travm = __DBL_MAX__, rd = 0.0f;
    //     xtmp = ix;
    //     ytmp = iy;
    //     ztmp = iz;
    //     if (ix == nnx)
    //         xtmp = nnx - 1;
    //     if (iy == nny)
    //         ytmp = nny - 1;
    //     if (iz == nnz)
    //         ztmp = nnz - 1;
    //     slown = 1.0f / _vel[xtmp * n_slices + ytmp * n_rows * n_slices + ztmp];
    //     if ((ix >= 0 && ix < nnx) && (iy >= 0 && iy < nny) && (iz >= 0 && iz < nnz))
    //     {
    //         if (ix - 2 < 0)
    //         {
    //             axx = 1;
    //         }
    //         if (ix + 2 >= nnx)
    //         {
    //             axx = -1;
    //         }

    //         if (iy - 2 < 0)
    //         {
    //             ayy = 1;
    //         }
    //         if (iy + 2 >= nny)
    //         {
    //             ayy = -1;
    //         }

    //         if (iz - 2 < 0)
    //         {
    //             azz = 1;
    //         }
    //         if (iz + 2 >= nnz)
    //         {
    //             azz = -1;
    //         }

    //         if (ix - 1 < 0)
    //         {
    //             ax = 1;
    //         }
    //         else if (ix + 1 >= nnx)
    //         {
    //             ax = -1;
    //         }
    //         else if (ttn[(ix + 1) * nnz + iy * nnx * nnz + iz] <=
    //                  ttn[(ix - 1) * nnz + iy * nnx * nnz + iz])
    //         {
    //             ax = 1;
    //         }
    //         else
    //         {
    //             ax = -1;
    //         }

    //         if (iy - 1 < 0)
    //         {
    //             ay = 1;
    //         }
    //         else if (iy + 1 >= nny)
    //         {
    //             ay = -1;
    //         }
    //         else if (ttn[ix * nnz + (iy + 1) * nnx * nnz + iz] <=
    //                  ttn[ix * nnz + (iy - 1) * nnx * nnz + iz])
    //         {
    //             ay = 1;
    //         }
    //         else
    //         {
    //             ay = -1;
    //         }

    //         if (iz - 1 < 0)
    //         {
    //             az = 1;
    //         }
    //         else if (iz + 1 >= nnz)
    //         {
    //             az = -1;
    //         }
    //         else if (ttn[ix * nnz + iy * nnx * nnz + iz + 1] <=
    //                  ttn[ix * nnz + iy * nnx * nnz + iz - 1])
    //         {
    //             az = 1;
    //         }
    //         else
    //         {
    //             az = -1;
    //         }

    //         ttnminx = ttn[(ix + ax) * nnz + iy * nnz * nnx + iz];
    //         ttnminy = ttn[ix * nnz + (iy + ay) * nnz * nnx + iz];
    //         ttnminz = ttn[ix * nnz + iy * nnz * nnx + iz + az];

    //         if (axx == 0)
    //             ttnminx2 = ttn[(ix + 2 * ax) * nnz + iy * nnz * nnx + iz];

    //         if (ayy == 0)
    //             ttnminy2 = ttn[ix * nnz + (iy + 2 * ay) * nnz * nnx + iz];

    //         if (azz == 0)
    //             ttnminz2 = ttn[ix * nnz + iy * nnz * nnx + iz + 2 * az];

    //         if (ttnminx <= ttnminy && ttnminy <= ttnminz)
    //         {
    //             c = ttnminx;
    //             b = ttnminy;
    //             a = ttnminz;
    //             c2 = ttnminx2;
    //             b2 = ttnminy2;
    //             a2 = ttnminz2;
    //         }
    //         else if (ttnminx <= ttnminz && ttnminz <= ttnminy)
    //         {
    //             c = ttnminx;
    //             b = ttnminz;
    //             a = ttnminy;
    //             c2 = ttnminx2;
    //             b2 = ttnminz2;
    //             a2 = ttnminy2;
    //         }
    //         else if (ttnminy <= ttnminz && ttnminz <= ttnminx)
    //         {
    //             c = ttnminy;
    //             b = ttnminz;
    //             a = ttnminx;
    //             c2 = ttnminy2;
    //             b2 = ttnminz2;
    //             a2 = ttnminx2;
    //         }
    //         else if (ttnminy <= ttnminx && ttnminx <= ttnminz)
    //         {
    //             c = ttnminy;
    //             b = ttnminx;
    //             a = ttnminz;
    //             c2 = ttnminy2;
    //             b2 = ttnminx2;
    //             a2 = ttnminz2;
    //         }
    //         else if (ttnminz <= ttnminx && ttnminx <= ttnminy)
    //         {
    //             c = ttnminz;
    //             b = ttnminx;
    //             a = ttnminy;
    //             c2 = ttnminz2;
    //             b2 = ttnminx2;
    //             a2 = ttnminy2;
    //         }
    //         else if (ttnminz <= ttnminy && ttnminy <= ttnminx)
    //         {
    //             c = ttnminz;
    //             b = ttnminy;
    //             a = ttnminx;
    //             c2 = ttnminz2;
    //             b2 = ttnminy2;
    //             a2 = ttnminx2;
    //         }

    //         trav = c + dnx * slown;

    //         if (trav > b)
    //         {
    //             trav11 = (b + c +
    //                       sqrtf(-b * b - c * c + 2.0f * b * c +
    //                             2.0f * dnx * dnx * (slown * slown))) /
    //                      2.0f;

    //             if (trav11 < trav)
    //             {
    //                 trav = trav11;
    //             }

    //             if (c2 < c)
    //             {
    //                 trav2 =
    //                     (3.0f * (4.0f * c - c2) + 4.0f * b +
    //                      2.0f * sqrtf(13 * slown * slown * dnx * dnx -
    //                                  (4.0f * c - c2 - 3.0f * b) * (4.0f * c - c2 - 3.0f * b))) /
    //                     13.0f;
    //                 if (trav2 < trav)
    //                 {
    //                     trav = trav2;
    //                 }
    //                 if (b2 < b)
    //                 {
    //                     trav3 = ((4.0f * c - c2) + (4.0f * b - b2) +
    //                              sqrtf(8.0f * (slown * slown * dnx * dnx) -
    //                                    ((4.0f * c - c2) - (4.0f * b - b2)) *
    //                                        ((4.0f * c - c2) - (4.0f * b - b2)))) /
    //                             6.0f;
    //                 }
    //                 if (trav3 < trav)
    //                 {
    //                     trav = trav3;
    //                 }
    //             }

    //             if (trav > a)
    //             {
    //                 trav = (2.0f * (a + b + c) +
    //                         sqrtf(4.0f * (a + b + c) * (a + b + c) -
    //                               12.0f * (a * a + b * b + c * c -
    //                                       1.0f * dnx * dnx * (slown * slown)))) /
    //                        6.0f;

    //                 if ((c2 > c))
    //                 {
    //                     trav_diag = F_(ttn,ix + ax) , (iy + ay) * nnz * nnx + iz + az] +
    //                                 sqrt(3.0f) * dnx * slown;

    //                     if (trav_diag < trav)
    //                     {
    //                         trav = trav_diag;
    //                     }
    //                 }

    //                 if (c2 < c)
    //                 {
    //                     aa1 = 9.0f / (4.0f * dnx * dnx);
    //                     bb1 = (-24.0f * c + 6.0f * c2) / (4.0f * dnx * dnx);
    //                     cc1 = ((4.0f * c - c2) * (4.0f * c - c2)) / (4.0f * dnx * dnx);

    //                     aa2 = 1.0f / (dny * dny);
    //                     bb2 = -2.0f * b / (dny * dny);
    //                     cc2 = b * b / (dny * dny);

    //                     aa3 = 1.0f / (dnz * dnz);
    //                     bb3 = -2.0f * a / (dnz * dnz);
    //                     cc3 = a * a / (dnz * dnz);

    //                     aa = aa1 + aa2 + aa3;
    //                     bb = bb1 + bb2 + bb3;
    //                     cc = cc1 + cc2 + cc3 - (slown * slown);

    //                     rd = bb * bb - 4.0f * aa * cc;
    //                     if (rd < 0.0f)
    //                         rd = 0.0f;
    //                     trav2 = (-bb + sqrt(rd)) / (2.0f * aa);

    //                     if (trav2 < trav)
    //                     {
    //                         trav = trav2;
    //                     }

    //                     if (b2 < b)
    //                     {
    //                         aa1 = 9.0f / (4.0f * dnx * dnx);
    //                         bb1 = (-24.0f * c + 6.0f * c2) / (4.0f * dnx * dnx);
    //                         cc1 = ((4.0f * c - c2) * (4.0f * c - c2)) / (4.0f * dnx * dnx);

    //                         aa2 = 9.0f / (4.0f * dny * dny);
    //                         bb2 = (-24.0f * b + 6.0f * b2) / (4.0f * dny * dny);
    //                         cc2 = ((4.0f * b - b2) * (4.0f * b - b2)) / (4.0f * dny * dny);

    //                         aa3 = 1.0f / (dnz * dnz);
    //                         bb3 = -2.0f * a / (dnz * dnz);
    //                         cc3 = a * a / (dnz * dnz);

    //                         aa = aa1 + aa2 + aa3;
    //                         bb = bb1 + bb2 + bb3;
    //                         cc = cc1 + cc2 + cc3 - (slown * slown);

    //                         rd = bb * bb - 4.0f * aa * cc;
    //                         if (rd < 0.0f)
    //                             rd = 0.0f;
    //                         trav3 = (-bb + sqrt(rd)) / (2.0f * aa);

    //                         if (trav3 < trav)
    //                         {
    //                             trav = trav3;
    //                         }

    //                         if (a2 < a)
    //                         {
    //                             aa1 = 9.0f / (4.0f * dnx * dnx);
    //                             bb1 = (-24.0f * c + 6.0f * c2) / (4.0f * dnx * dnx);
    //                             cc1 = ((4.0f * c - c2) * (4.0f * c - c2)) / (4.0f * dnx * dnx);

    //                             aa2 = 9.0f / (4.0f * dny * dny);
    //                             bb2 = (-24.0f * b + 6.0f * b2) / (4.0f * dny * dny);
    //                             cc2 = ((4.0f * b - b2) * (4.0f * b - b2)) / (4.0f * dny * dny);

    //                             aa3 = 9.0f / (4.0f * dnz * dnz);
    //                             bb3 = (-24.0f * a + 6.0f * a2) / (4.0f * dnz * dnz);
    //                             cc3 = ((4.0f * a - a2) * (4.0f * a - a2)) / (4.0f * dnz * dnz);

    //                             aa = aa1 + aa2 + aa3;
    //                             bb = bb1 + bb2 + bb3;
    //                             cc = cc1 + cc2 + cc3 - (slown * slown);

    //                             rd = bb * bb - 4.0f * aa * cc;
    //                             if (rd < 0.0f)
    //                                 rd = 0.0f;
    //                             trav4 = (-bb + sqrt(rd)) / (2.0f * aa);

    //                             if (trav4 < trav)
    //                             {
    //                                 trav = trav4;
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //         travm = trav;
    //     }
    //     return travm;
    // }

    __device__ double diff3d_2diag(int ix, int iy, int iz, Frame RayVelGrid, float *_vel, double *_time)
    {
        SET_MODEL_GRID_ND(RayVelGrid);
        double a, b, c, slown;
        double a2, b2, c2;
        double aa, bb, cc;
        double aa1, bb1, cc1;
        double aa2, bb2, cc2;
        double aa3, bb3, cc3;
        double trav, trav1, trav2, trav3, trav4, trav5, trav6, trav7, trav_d, travm, rd;
        int xtmp, ytmp, ztmp;
        double ttnminx, ttnminy, ttnminz;
        double ttnminx2, ttnminy2, ttnminz2;
        ttnminx2 = __DBL_MAX__, ttnminy2 = __DBL_MAX__, ttnminz2 = __DBL_MAX__;
        int ax, ay, az;

        double trav_diag = __DBL_MAX__;
        trav = 0.0f, trav1 = __DBL_MAX__, trav2 = __DBL_MAX__, trav3 = __DBL_MAX__,
        trav4 = __DBL_MAX__, trav5 = __DBL_MAX__, trav6 = __DBL_MAX__, trav7 = __DBL_MAX__, trav_d = __DBL_MAX__,
        travm = __DBL_MAX__, rd = 0.0f;
        xtmp = ix;
        ytmp = iy;
        ztmp = iz;
        if (ix == n_rows)
            xtmp = n_rows - 1;
        if (iy == n_cols)
            ytmp = n_cols - 1;
        if (iz == n_slices)
            ztmp = n_slices - 1;
        slown = 1.0f / F_(_vel, xtmp, ytmp, ztmp);
        if ((ix >= 0 && ix < n_rows) && (iy >= 0 && iy < n_cols) && (iz >= 0 && iz < n_slices))
        {
            if (ix - 1 < 0)
                ax = 1;
            else if (ix + 1 >= n_rows)
                ax = -1;
            else if (F_(_time, (ix + 1), iy, iz) <= F_(_time, (ix - 1), iy, iz))
                ax = 1;
            else
                ax = -1;

            if (iy - 1 < 0)
                ay = 1;
            else if (iy + 1 >= n_cols)
                ay = -1;
            else if (F_(_time, ix, (iy + 1), iz) <= F_(_time, ix, (iy - 1), iz))
                ay = 1;
            else
                ay = -1;

            if (iz - 1 < 0)
                az = 1;
            else if (iz + 1 >= n_slices)
                az = -1;
            else if (F_(_time, ix, iy, (iz + 1)) <= F_(_time, ix, iy, (iz - 1)))
                az = 1;
            else
                az = -1;

            ttnminx = F_(_time, (ix + ax), iy, iz);
            ttnminy = F_(_time, ix, (iy + ay), iz);
            ttnminz = F_(_time, ix, iy, (iz + az));

            ttnminx2 = F_(_time, (ix + 2 * ax), iy, iz);
            ttnminy2 = F_(_time, ix, (iy + 2 * ay), iz);
            ttnminz2 = F_(_time, ix, iy, (iz + 2 * az));

            int p[3] = {0, 1, 2}; //用于123全排列
            double ttnmin[3] = {ttnminx, ttnminy, ttnminz};
            double ttnmin2[3] = {ttnminx2, ttnminy2, ttnminz2};
            do
            {
                if (ttnmin[p[0]] <= ttnmin[p[1]] && ttnmin[p[1]] <= ttnmin[p[2]])
                {
                    c = ttnmin[p[0]];
                    b = ttnmin[p[1]];
                    a = ttnmin[p[2]];
                    c2 = ttnmin2[p[0]];
                    b2 = ttnmin2[p[1]];
                    a2 = ttnmin2[p[2]];
                }
            } while (CudaNextPermutation(p, 3));

            trav = c + d_rows * slown;

            if (trav > b)
            {
                trav = (b + c +
                        sqrt(-b * b - c * c + 2.0f * b * c +
                             2.0f * d_rows * d_rows * (slown * slown))) /
                       2.0f;

                if (c2 < c)
                {
                    trav1 =
                        (3.0f * (4.0f * c - c2) + 4.0f * b +
                         2.0f * sqrt(13 * slown * slown * d_rows * d_rows -
                                     (4.0f * c - c2 - 3.0f * b) * (4.0f * c - c2 - 3.0f * b))) /
                        13.0f;

                    if (b2 < b)
                    {
                        trav2 = ((4.0f * c - c2) + (4.0f * b - b2) +
                                 sqrt(8.0f * (slown * slown * d_rows * d_rows) -
                                      ((4.0f * c - c2) - (4.0f * b - b2)) * ((4.0f * c - c2) - (4.0f * b - b2)))) /
                                6.0f;
                    }
                }
                //
                if (trav1 < trav)
                {
                    trav = trav1;
                }
                if (trav2 < trav)
                {
                    trav = trav2;
                }
                //
                if (trav > a)
                {
                    trav =
                        (2.0f * (a + b + c) + sqrt(4.0f * (a + b + c) * (a + b + c) -
                                                   12.0f * (a * a + b * b + c * c -
                                                            d_rows * d_rows * (slown * slown)))) /
                        6.0f;

                    if (c2 < c)
                    {
                        aa1 = 9.0f / (4.0f * d_rows * d_rows);
                        bb1 = (-24.0f * c + 6.0f * c2) / (4.0f * d_rows * d_rows);
                        cc1 = ((4.0f * c - c2) * (4.0f * c - c2)) / (4.0f * d_rows * d_rows);

                        aa2 = 1.0f / (d_cols * d_cols);
                        bb2 = -2.0f * b / (d_cols * d_cols);
                        cc2 = b * b / (d_cols * d_cols);

                        aa3 = 1.0f / (d_slices * d_slices);
                        bb3 = -2.0f * a / (d_slices * d_slices);
                        cc3 = a * a / (d_slices * d_slices);

                        aa = aa1 + aa2 + aa3;
                        bb = bb1 + bb2 + bb3;
                        cc = cc1 + cc2 + cc3 - (slown * slown);

                        rd = bb * bb - 4.0f * aa * cc;
                        if (rd < 0.0f)
                            rd = 0.0f;
                        trav3 = (-bb + sqrt(rd)) / (2.0f * aa);

                        if (b2 < b)
                        {
                            aa1 = 9.0f / (4.0f * d_rows * d_rows);
                            bb1 = (-24.0f * c + 6.0f * c2) / (4.0f * d_rows * d_rows);
                            cc1 = ((4.0f * c - c2) * (4.0f * c - c2)) / (4.0f * d_rows * d_rows);
                            aa2 = 9.0f / (4.0f * d_cols * d_cols);
                            bb2 = (-24.0f * b + 6.0f * b2) / (4.0f * d_cols * d_cols);
                            cc2 = ((4.0f * b - b2) * (4.0f * b - b2)) / (4.0f * d_cols * d_cols);

                            aa3 = 1.0f / (d_slices * d_slices);
                            bb3 = -2.0f * a / (d_slices * d_slices);
                            cc3 = a * a / (d_slices * d_slices);

                            aa = aa1 + aa2 + aa3;
                            bb = bb1 + bb2 + bb3;
                            cc = cc1 + cc2 + cc3 - (slown * slown);
                            rd = bb * bb - 4.0f * aa * cc;
                            if (rd < 0.0f)
                                rd = 0.0f;
                            trav4 = (-bb + sqrt(rd)) / (2.0f * aa);

                            if (a2 < a)
                            {
                                aa1 = 9.0f / (4.0f * d_rows * d_rows);
                                bb1 = (-24.0f * c + 6.0f * c2) / (4.0f * d_rows * d_rows);
                                cc1 = ((4.0f * c - c2) * (4.0f * c - c2)) / (4.0f * d_rows * d_rows);
                                aa2 = 9.0f / (4.0f * d_cols * d_cols);
                                bb2 = (-24.0f * b + 6.0f * b2) / (4.0f * d_cols * d_cols);
                                cc2 = ((4.0f * b - b2) * (4.0f * b - b2)) / (4.0f * d_cols * d_cols);
                                aa3 = 9.0f / (4.0f * d_slices * d_slices);
                                bb3 = (-24.0f * a + 6.0f * a2) / (4.0f * d_slices * d_slices);
                                cc3 = ((4.0f * a - a2) * (4.0f * a - a2)) / (4.0f * d_slices * d_slices);
                                aa = aa1 + aa2 + aa3;
                                bb = bb1 + bb2 + bb3;
                                cc = cc1 + cc2 + cc3 - (slown * slown);
                                rd = bb * bb - 4.0f * aa * cc;
                                if (rd < 0.0f)
                                    rd = 0.0f;
                                trav5 = (-bb + sqrt(rd)) / (2.0f * aa);
                                //
                                trav_diag = F_(_time, ix + ax, iy + ay, iz + az) + sqrt(3.0f) * d_rows * slown;
                            }
                        }
                    }
                    //
                    if (trav_diag < trav)
                    {
                        trav = trav_diag;
                    }
                    if (trav3 < trav)
                    {
                        trav = trav3;
                    }
                    if (trav4 < trav)
                    {
                        trav = trav4;
                    }
                    if (trav5 < trav)
                    {
                        trav = trav5;
                    }
                    //
                }
            }
            travm = trav;
        }
        return travm;
    }
}
#endif