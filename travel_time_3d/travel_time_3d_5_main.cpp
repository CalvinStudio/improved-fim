#ifndef TRAVEL_TIME_3D_MAIN_CPP
#define TRAVEL_TIME_3D_MAIN_CPP
// #define RACC_DEBUG
#include "travel_time_3d_0.hpp"
using namespace racc;
int main(int argc, char *argv[])
{
    if (!argv[1])
    {
        cout << "Please input argfile!" << endl;
        RACC_ERROR_EXIT;
    }
    string path = string(argv[1]) + "/model/cube200.gms";
#ifdef RACC_DEBUG
    cout << "DEBUG MODEL" << endl;
#endif
    //
    cudaSetDevice(1);
    travel_time_3d_module TT3Dthe;
    travel_time_3d_module TT3D11;
    travel_time_3d_module TT3D12;
    travel_time_3d_module TT3D21;
    travel_time_3d_module TT3D22;
    cout << path << endl;
    CGmsReader gms(path);
    //
    ffield _vel("_vel");
    gms.ReadGMSByOrderToField(_vel);
    //
    _vel.frame.PrintInfo();
    fcufld Vel(MemType::pin, _vel.frame);
    Vel.copy(_vel);
    tp3vec shot(1);
    shot(0).x = 100;
    shot(0).y = 100;
    shot(0).z = 100;
    shot(0).time = 0;
    //
    tp3vec rece(1);
    rece(0).x = 25;
    rece(0).y = 50;
    rece(0).z = 20;
    //
    TT3D11.init(Vel, 11);
    TT3D12.init(Vel, 12);
    TT3D21.init(Vel, 21);
    TT3D22.init(Vel, 22);
    TT3Dthe.init(Vel, 11);
    //
    shot.print_struct();
    TIC(1);

    TIC(33);
    TT3Dthe.set_source(shot);
    TT3Dthe.cal_true_time();
    TOC(33, "33:");

    // for (int ie = 1; ie <= 1; ie++)
    // {
    //     for (int id = 2; id <= 2; id++)
    //     {
    TIC(22);
    // TT3D22.set_source(shot);
    // TT3D22.cu_cal_time();
    TT3D22.cal_refine_time(shot, 10, 5);
    TOC(22, "COST TIME:");

    // TT3D11.get_dev_time();
    // TT3D12.get_dev_time();
    // TT3D22.get_dev_time();
    TT3D22.get_dev_time();
    //
    field time_error(TT3D22.time.frame);
    //
    for (int i = 0; i < TT3D22.time.frame.n_elem; i++)
    {
        time_error[i] = TT3D22.time[i] - TT3Dthe.time[i];
    }
    //
    time_error.save("/home/calvin/WKSP/INCLUDE/raccoon/app/cpp/travel_time_3d/demo/out/timeerror21refine" + TT3D22.time.size_str(), SaveFormat::binary_raw);
    cudaDeviceSynchronize();
    TT3D22.time.save("/home/calvin/WKSP/INCLUDE/raccoon/app/cpp/travel_time_3d/demo/out/time_r" + TT3D22.time.size_str(), SaveFormat::binary_raw);

    cout << "time_error_mean:  " << time_error.mean() << endl;
    cout << "time_error_max :  " << time_error.max() << endl;
    cout << "time_error_min :  " << time_error.min() << endl;
    // }
    // }

    TOC(1, "ALL TIME:");
    return 0;
}

#endif