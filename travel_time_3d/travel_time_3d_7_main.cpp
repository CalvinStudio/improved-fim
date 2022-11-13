#pragma once
#ifndef TRAVEL_TIME_3D_MAIN_CPP
#define TRAVEL_TIME_3D_MAIN_CPP
#include "travel_time_3d_5_meat.hpp"
#include "include/.jarvis_2_gms/.jarvis_device_gms_header_out.h"
using namespace jarvis;
int main(int argc, char *argv[])
{
    if (!argv[1])
    {
        cout << "Please input argfile!" << endl;
        std::abort();
    }
    string path = string(argv[1]) + "/model/cube200.gms";
#ifdef JARVIS_DEBUG
    cout << "DEBUG MODEL" << endl;
#endif
    //
    cudaSetDevice(1);
    travel_time_3d_module travel_time_3d_theoretical;
    travel_time_3d_module travel_time_3d_1rd;
    travel_time_3d_module travel_time_3d_1rd_diag;
    travel_time_3d_module travel_time_3d_2rd;
    travel_time_3d_module travel_time_3d_2rd_diag;
    cout << path << endl;
    jarvis::GmsReader gms(path);
    //
    fcufld Vel;
    gms.read_gms_by_order_to_fcufield(Vel);
    //
    Vel.frame.print_info();
    tp3cuvec shot(MemType::pin, 1);
    shot(0).x = 100;
    shot(0).y = 100;
    shot(0).z = 100;
    shot(0).time = 0;
    //
    tp3cuvec rece(MemType::pin, 1);
    rece(0).x = 25;
    rece(0).y = 50;
    rece(0).z = 20;
    //
    travel_time_3d_1rd.module_init(&Vel, 11);
    travel_time_3d_1rd_diag.module_init(&Vel, 12);
    travel_time_3d_2rd.module_init(&Vel, 21);
    travel_time_3d_2rd_diag.module_init(&Vel, 22);
    travel_time_3d_theoretical.module_init(&Vel, 11);
    //
    travel_time_3d_theoretical.cal_travel_time(shot, TravelType::theoretical);
    TIC(0);
    travel_time_3d_1rd.cal_travel_time(shot, TravelType::normal);
    TOC(0, "COST TIME1:");

    TIC(1);
    travel_time_3d_1rd_diag.cal_travel_time(shot, TravelType::normal);
    TOC(1, "COST TIME2:");

    TIC(2);
    travel_time_3d_2rd.cal_travel_time(shot, TravelType::normal);
    TOC(2, "COST TIME3:");

    TIC(3);
    travel_time_3d_2rd_diag.cal_travel_time(shot, TravelType::normal);
    TOC(3, "COST TIME4:");

    travel_time_3d_1rd.get_device_time();
    travel_time_3d_1rd_diag.get_device_time();
    travel_time_3d_2rd.get_device_time();
    travel_time_3d_2rd_diag.get_device_time();
    field time_error(travel_time_3d_2rd_diag.time.frame);
    for (int i = 0; i < travel_time_3d_1rd.time.frame.n_elem; i++)
    {
        time_error[i] = travel_time_3d_1rd.time[i] - travel_time_3d_theoretical.time[i];
    }
    cout << "travel_time_3d_1rd:time_error_mean:  " << time_error.mean() << endl;
    cout << "travel_time_3d_1rd:time_error_max :  " << time_error.max() << endl;
    cout << "travel_time_3d_1rd:time_error_min :  " << time_error.min() << endl;
    cout << endl;
    //
    for (int i = 0; i < travel_time_3d_1rd_diag.time.frame.n_elem; i++)
    {
        time_error[i] = travel_time_3d_1rd_diag.time[i] - travel_time_3d_theoretical.time[i];
    }
    cout << "travel_time_3d_1rd_diag:time_error_mean:  " << time_error.mean() << endl;
    cout << "travel_time_3d_1rd_diag:time_error_max :  " << time_error.max() << endl;
    cout << "travel_time_3d_1rd_diag:time_error_min :  " << time_error.min() << endl;
    cout << endl;
    //
    for (int i = 0; i < travel_time_3d_2rd.time.frame.n_elem; i++)
    {
        time_error[i] = travel_time_3d_2rd.time[i] - travel_time_3d_theoretical.time[i];
    }
    cout << "travel_time_3d_2rd:time_error_mean:  " << time_error.mean() << endl;
    cout << "travel_time_3d_2rd:time_error_max :  " << time_error.max() << endl;
    cout << "travel_time_3d_2rd:time_error_min :  " << time_error.min() << endl;
    cout << endl;
    //
    for (int i = 0; i < travel_time_3d_2rd_diag.time.frame.n_elem; i++)
    {
        time_error[i] = travel_time_3d_2rd_diag.time[i] - travel_time_3d_theoretical.time[i];
    }
    cout << "travel_time_3d_2rd_diag:time_error_mean:  " << time_error.mean() << endl;
    cout << "travel_time_3d_2rd_diag:time_error_max :  " << time_error.max() << endl;
    cout << "travel_time_3d_2rd_diag:time_error_min :  " << time_error.min() << endl;
    cout << endl;
    return 0;
}
#endif