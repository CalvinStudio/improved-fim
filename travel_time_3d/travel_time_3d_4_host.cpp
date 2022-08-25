#ifndef TRAVEL_TIME_3D_CPP
#define TRAVEL_TIME_3D_CPP
#include "travel_time_3d_0.hpp"
namespace racc
{
    void travel_time_3d_module::init(const fcufld &_vel, int _diff_order)
    {
#ifdef RACC_DEBUG
        if (_diff_order != 11 && _diff_order != 12 && _diff_order != 21 && _diff_order != 22)
        {
            cout << "travel_time_3d_module::Init():diff_order= 11 or 12 or 21 or 22!" << endl;
            RACC_ERROR_EXIT;
        }
#endif
        vel = _vel;
        diff_order = _diff_order;
        vel.set_name("vel");
        time.set_name("time");
        mark.set_name("mark");
        time.cu_alloc(MemType::pin, _vel.frame);
        mark.cu_alloc(MemType::pin, _vel.frame);
        vel.cu_stream_copy_h2d(racc_default_stream);
        endflag.cu_alloc(MemType::pin, 1);
        Blocks = (vel.frame.n_elem + racc_const::block_size - 1) / racc_const::block_size;
        Threads = racc_const::block_size;
    }

    dcufld travel_time_3d_module::GetTravelTimeField()
    {
#ifdef RACC_DEBUG
        if (&time[0])
        {
#endif
            return time;
#ifdef RACC_DEBUG
        }
        else if (&time[0] && !time[0])
        {
            std::cout << "travel_time_3d_module::GetTravelTimeField():[ERROR]:The time of memory has been released by the copied object !" << std::endl;
            RACC_ERROR_EXIT;
        }
        else
        {
            std::cout << "travel_time_3d_module::GetTravelTimeField():[ERROR]:travel_time_3d_module Error!" << std::endl;
            RACC_ERROR_EXIT;
        }
#endif
    }

    void travel_time_3d_module::get_dev_time()
    {
        time.cu_stream_copy_d2h(racc_default_stream);
        cudaDeviceSynchronize();
    }

    void travel_time_3d_module::get_time_at(tp3vec &position)
    {
        _GetTimeAt(time, position);
    }

    void travel_time_3d_module::cal_true_time()
    {
        SET_MODEL_GRID_NDL(vel.frame);
        float sx, sy, sz, ds;
        for (int k = 0; k < n_slices; k++)
            for (int j = 0; j < n_cols; j++)
                for (int i = 0; i < n_rows; i++)
                {
                    sx = l_rows + i * d_rows - shotline(0).x;
                    sy = l_cols + j * d_cols - shotline(0).y;
                    sz = l_slices + k * d_slices - shotline(0).z;
                    ds = sqrt(sx * sx + sy * sy + sz * sz);
                    time(i, j, k) = ds / vel(0, 0, 0);
                }
    }

    void travel_time_3d_module::del_time()
    {
        time.del();
    }

    void travel_time_3d_module::set_source(tp3vec &_shotline)
    {
        SET_MODEL_GRID_NDL(vel.frame);
        CheckShotLine(vel.frame, _shotline);
        shotline = _shotline;
        mark.fill(-1);
        time.fill(__DBL_MAX__);
        float sx, sy, sz, ds;
        float ishotx, ishoty, ishotz;
        int isx, isy, isz;
        for (int ishot = 0; ishot < shotline.n_elem; ishot++)
        {
            isx = int((shotline(ishot).x - l_rows) / d_rows);
            isy = int((shotline(ishot).y - l_cols) / d_cols);
            isz = int((shotline(ishot).z - l_slices) / d_slices);
            // ishotx = int((100 - l_rows) / d_rows);
            // ishoty = int((100 - l_cols) / d_cols);
            // ishotz = int((100 - l_slices) / d_slices);
            // if (abs(isx - ishotx) + abs(isy - ishoty) + abs(isz - ishotz) < 10)
            {
                mark(isx, isy, isz) = 3; //加密网格标记
                time(isx, isy, isz) = shotline(ishot).time;
                //
                int lax = -1;
                int rax = 1;
                int lay = -1;
                int ray = 1;
                int laz = -1;
                int raz = 1;

                if (isx <= 0)
                    lax = 1;
                if (isx >= n_rows - 1)
                    rax = -1;

                if (isy <= 0)
                    lay = 1;
                if (isy >= n_cols - 1)
                    ray = -1;

                if (isz <= 0)
                    laz = 1;
                if (isz >= n_slices - 1)
                    raz = -1;

                if (mark(isx + lax, isy, isz) != 3)
                    mark(isx + lax, isy, isz) = 1;

                if (mark(isx + rax, isy, isz) != 3)
                    mark(isx + rax, isy, isz) = 1;

                if (mark(isx, isy + lay, isz) != 3)
                    mark(isx, isy + lay, isz) = 1;

                if (mark(isx, isy + ray, isz) != 3)
                    mark(isx, isy + ray, isz) = 1;

                if (mark(isx, isy, isz + laz) != 3)
                    mark(isx, isy, isz + laz) = 1;

                if (mark(isx, isy, isz + raz) != 3)
                    mark(isx, isy, isz + raz) = 1;
            }
        }
        time.cu_stream_copy_h2d(racc_default_stream);
        mark.cu_stream_copy_h2d(racc_default_stream);
        cudaDeviceSynchronize();
    }

    inline void GetSubGridValue(const fcufld &a, fcufld &sub_a)
    {
        SET_MODEL_GRID_NDL(a.frame);
        long sub_mnx = sub_a.frame.n_rows, sub_mnz = sub_a.frame.n_cols;
        double sub_mdx = sub_a.frame.d_rows, sub_mdz = sub_a.frame.d_cols;
        double sub_mlx = sub_a.frame.l_rows, sub_mlz = sub_a.frame.l_cols;
        int isx, isy, isz;
        double sx, sy, sz;
        for (int k = 0; k < sub_a.frame.n_slices; k++)
            for (int j = 0; j < sub_a.frame.n_cols; j++)
                for (int i = 0; i < sub_a.frame.n_rows; i++)
                {
                    double posx = sub_a.frame.l_rows + sub_a.frame.d_rows * i;
                    double posy = sub_a.frame.l_cols + sub_a.frame.d_cols * j;
                    double posz = sub_a.frame.l_slices + sub_a.frame.d_slices * k;

                    isx = int((posx - l_rows) / d_rows);
                    isy = int((posy - l_cols) / d_cols);
                    isz = int((posz - l_slices) / d_slices);
                    if (isx == n_rows)
                        isx = n_rows - 1;
                    if (isx >= n_rows - 1)
                        isx = n_rows - 2;
                    //
                    if (isy == n_cols)
                        isy = n_cols - 1;
                    if (isy >= n_cols - 1)
                        isy = n_cols - 2;
                    //
                    if (isz == n_slices)
                        isz = n_slices - 1;
                    if (isz >= n_slices - 1)
                        isz = n_slices - 2;

                    dvec vec(3);
                    vec(0) = double(a(isx + 1, isy, isz)) - a(isx, isy, isz);
                    vec(1) = double(a(isx, isy + 1, isz)) - a(isx, isy, isz);
                    vec(2) = double(a(isx, isy, isz + 1)) - a(isx, isy, isz);
                    sx = posx - d_rows * isx;
                    sy = posy - d_cols * isy;
                    sz = posz - d_slices * isz;
                    double dis = (sx * vec(0) + sy * vec(1) + sz * vec(2));
                    sub_a(i, j, k) = float(dis / d_rows + a(isx, isz));
                }
    }

    void travel_time_3d_module::cal_refine_time(tp3vec &_shotline, int extend_num, int divide_num)
    {
        TIC(0);
#ifdef RACC_DEBUG
        if (_shotline.n_elem == 1 && extend_num > 0 && divide_num > 0)
        {
#endif
            double lbx = _shotline(0).x - extend_num * vel.frame.d_rows;
            double rbx = _shotline(0).x + extend_num * vel.frame.d_rows;
            double lby = _shotline(0).y - extend_num * vel.frame.d_cols;
            double rby = _shotline(0).y + extend_num * vel.frame.d_cols;
            double lbz = _shotline(0).z - extend_num * vel.frame.d_slices;
            double rbz = _shotline(0).z + extend_num * vel.frame.d_slices;

            if (lbx < vel.frame.l_rows)
                lbx = vel.frame.l_rows;
            if (rbx > vel.frame.r_rows)
                rbx = vel.frame.r_rows;
            //
            if (lby < vel.frame.l_cols)
                lby = vel.frame.l_cols;
            if (rby > vel.frame.r_cols)
                rby = vel.frame.r_cols;
            //
            if (lbz < vel.frame.l_slices)
                lbz = vel.frame.l_slices;
            if (rbz > vel.frame.r_slices)
                rbz = vel.frame.r_slices;
            //
            Frame refine_grid;
            refine_grid.l_rows = lbx;
            refine_grid.r_rows = rbx;
            refine_grid.l_cols = lby;
            refine_grid.r_cols = rby;
            refine_grid.l_slices = lbz;
            refine_grid.r_slices = rbz;
            //
            refine_grid.d_rows = vel.frame.d_rows / divide_num;
            refine_grid.d_cols = vel.frame.d_cols / divide_num;
            refine_grid.d_slices = vel.frame.d_slices / divide_num;
            refine_grid.n_rows = long((refine_grid.r_rows - refine_grid.l_rows) / refine_grid.d_rows + 1);
            refine_grid.n_cols = long((refine_grid.r_cols - refine_grid.l_cols) / refine_grid.d_cols + 1);
            refine_grid.n_slices = long((refine_grid.r_slices - refine_grid.l_slices) / refine_grid.d_slices + 1);
            //
            refine_grid.PrintInfo("refine_grid:");
            cout << diff_order << endl;
            fcufld refinevel;
            refinevel.cu_alloc(MemType::pin, refine_grid);
            GetSubGridValue(vel, refinevel);
            // refinevel.save("/home/calvin/WKSP/INCLUDE/raccoon/app/cpp/travel_time_3d/demo/out/refinevel" + refinevel.size_str(), SaveFormat::binary_raw);
            //
            travel_time_3d_module travelrefine;
            travelrefine.init(refinevel, diff_order);
            travelrefine.set_source(_shotline);
            travelrefine.cu_cal_time();
            travelrefine.get_dev_time();
            travelrefine.time.save("/home/calvin/WKSP/INCLUDE/raccoon/app/cpp/travel_time_3d/demo/out/refinetime" + travelrefine.time.size_str(), SaveFormat::binary_raw);
            //
            tp3vec refineline(((refine_grid.n_rows - 1) / divide_num + 1) *
                              ((refine_grid.n_cols - 1) / divide_num + 1) *
                              ((refine_grid.n_slices - 1) / divide_num + 1));
            //
            int linenum = 0;
            for (int k = 0; k < refine_grid.n_slices; k = k + divide_num)
                for (int j = 0; j < refine_grid.n_cols; j = j + divide_num)
                    for (int i = 0; i < refine_grid.n_rows; i = i + divide_num)
                    {
                        refineline(linenum).x = refine_grid.l_rows + refine_grid.d_rows * i;
                        refineline(linenum).y = refine_grid.l_cols + refine_grid.d_cols * j;
                        refineline(linenum).z = refine_grid.l_slices + refine_grid.d_slices * k;
                        refineline(linenum).time = travelrefine.time(i, j, k);
                        linenum++;
                    }
            refineline.print_struct();
            set_source(refineline);
            cu_cal_time();
            //
#ifdef RACC_DEBUG
        }
        else if (_shotline.n_elem > 1)
        {
            cout << "travel_time_3d_module::RefineTravel():[ERROR]:This function is only applicable to point sources!";
            RACC_ERROR_EXIT;
        }
        else if (extend_num <= 0)
        {
            cout << "travel_time_3d_module::RefineTravel:[ERROR]:Error extend_num!" << endl;
            RACC_ERROR_EXIT;
        }
        else if (divide_num <= 0)
        {
            cout << "travel_time_3d_module::RefineTravel:[ERROR]:Error divide_num!" << endl;
            RACC_ERROR_EXIT;
        }
        else
        {
            cout << "travel_time_3d_module::RefineTravel:[ERROR]:Error !" << endl;
            RACC_ERROR_EXIT;
        }
#endif
    }
}
#endif