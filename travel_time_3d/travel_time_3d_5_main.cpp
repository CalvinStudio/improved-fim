#pragma once
#ifndef TRAVEL_TIME_3D_MAIN_CPP
#define TRAVEL_TIME_3D_MAIN_CPP
#include "travel_time_3d_3_meat.hpp"
using namespace jarvis;
int main()
{
    cudaSetDevice(0);
    //
    travel_time_3d_module travel_time_3d_theoretical; //!(just for constant velocity model)To generate theoretical values
    travel_time_3d_module travel_time_3d_1rd;         //*First-order difference
    travel_time_3d_module travel_time_3d_1rd_diag;    //*first-order difference with diagonal node
    travel_time_3d_module travel_time_3d_2rd;         //*second-order difference
    travel_time_3d_module travel_time_3d_2rd_diag;    //*second-order difference with diagonal node

    //! input DIY velocity
    Frame vel_grid;
    vel_grid.set_nd(201, 201, 201, 1, 1, 1); //* 200 represets grid size;1 is grid node spacing
    fcufld Vel(MemType::npin, vel_grid);
    for (int k = 0; k < 201; k++)
        for (int j = 0; j < 201; j++)
            for (int i = 0; i < 201; i++)
                Vel(i, j, k) = 2000;
    Vel.cu_copy_h2d();

    Vel.frame.print_info(); //*print grid information
    tp3cuvec source(MemType::pin, 1);
    source(0).x = 100; //*source location
    source(0).y = 100;
    source(0).z = 100;
    source(0).time = 0;//*Source time is 0
    //
    travel_time_3d_1rd.module_init(&Vel, 11);//11:First-order difference
    travel_time_3d_1rd_diag.module_init(&Vel, 12);//12:first-order difference with diagonal node
    travel_time_3d_2rd.module_init(&Vel, 21);//21:second-order difference
    travel_time_3d_2rd_diag.module_init(&Vel, 22);//second-order difference with diagonal node
    travel_time_3d_theoretical.module_init(&Vel, 11);//11 Not required here
    //
    travel_time_3d_theoretical.cal_travel_time(source, TravelType::theoretical);//calculate theoretical values
    TIC(0);
    travel_time_3d_1rd.cal_travel_time(source, TravelType::normal);
    TOC(0, "First-order difference COST TIME1:");

    TIC(1);
    travel_time_3d_1rd_diag.cal_travel_time(source, TravelType::normal);
    TOC(1, "first-order difference with diagonal node COST TIME2:");

    TIC(2);
    travel_time_3d_2rd.cal_travel_time(source, TravelType::normal);
    TOC(2, "second-order difference COST TIME3:");

    TIC(3);
    travel_time_3d_2rd_diag.cal_travel_time(source, TravelType::normal);
    TOC(3, "second-order difference with diagonal node COST TIME4:");

    travel_time_3d_1rd.get_device_time();//Copy the travel time data on the GPU to the host memory
    travel_time_3d_1rd_diag.get_device_time();
    travel_time_3d_2rd.get_device_time();
    travel_time_3d_2rd_diag.get_device_time();
    field time_error(travel_time_3d_2rd_diag.time.frame);//calculation error
    for (int i = 0; i < travel_time_3d_1rd.time.frame.n_elem; i++)
    {
        time_error[i] = travel_time_3d_1rd.time[i] - travel_time_3d_theoretical.time[i];
    }
    cout << "travel_time_3d_1rd:time_error_mean:  " << time_error.mean() << endl;
    cout << "travel_time_3d_1rd:time_error_max :  " << time_error.max() << endl;
    cout << "travel_time_3d_1rd:time_error_min :  " << time_error.min() << endl; //*Very small negative value, can be regarded as zero.
    cout << endl;
    //
    for (int i = 0; i < travel_time_3d_1rd_diag.time.frame.n_elem; i++)
    {
        time_error[i] = travel_time_3d_1rd_diag.time[i] - travel_time_3d_theoretical.time[i];
    }
    cout << "travel_time_3d_1rd_diag:time_error_mean:  " << time_error.mean() << endl;
    cout << "travel_time_3d_1rd_diag:time_error_max :  " << time_error.max() << endl;
    cout << "travel_time_3d_1rd_diag:time_error_min :  " << time_error.min() << endl; //*Very small negative value, can be regarded as zero.
    cout << endl;
    //
    for (int i = 0; i < travel_time_3d_2rd.time.frame.n_elem; i++)
    {
        time_error[i] = travel_time_3d_2rd.time[i] - travel_time_3d_theoretical.time[i];
    }
    cout << "travel_time_3d_2rd:time_error_mean:  " << time_error.mean() << endl;
    cout << "travel_time_3d_2rd:time_error_max :  " << time_error.max() << endl;
    cout << "travel_time_3d_2rd:time_error_min :  " << time_error.min() << endl; //*Very small negative value, can be regarded as zero.
    cout << endl;
    //
    for (int i = 0; i < travel_time_3d_2rd_diag.time.frame.n_elem; i++)
    {
        time_error[i] = travel_time_3d_2rd_diag.time[i] - travel_time_3d_theoretical.time[i];
    }
    cout << "travel_time_3d_2rd_diag:time_error_mean:  " << time_error.mean() << endl;
    cout << "travel_time_3d_2rd_diag:time_error_max :  " << time_error.max() << endl;
    cout << "travel_time_3d_2rd_diag:time_error_min :  " << time_error.min() << endl; //*Very small negative value, can be regarded as zero.
    cout << endl;
    return 0;
}
#endif