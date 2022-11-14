<img src="./jarvis_min.jpg" width = "360" align=center />

# open_travel_time_3d_gpu

## Background
Traveltime calculation is widely used in seismic tomography,migration imaging, earthquake location, etc Fast Iterative Method(FIM) is one of the fastest methods for traveltime calculation since it can be easily accelerated by running in parallel, but its accuracy is poor and it cannot satisfy actual demands.Although the computational accuracy of FIM can be improved simply by increasing the grid density, this results in a rapid increase in computational workload and a significant reduction in computational efficiency. This study incorporates the high-order difference schemes, diagonal node calculation, and double-grid technique into the FIM to address this issue. The improved FIM can substantially increase the computational accuracy while essentially maintaining its original computational efficiency.

## **Key Points**
1. A novel FIM based on higher-order finite differences can considerably improve the accuracy of numerical solution of Eikonal equation.

2. Introducing diagonal node calculation and the double-grid technique into FIM can further improve calculation accuracy of the solution.

3. The improved FIM has a smaller increase in computational complexity, compared to the increase in accuracy.

## Software testing environment
### linux (Author's test environment:CentOS)
>g++ -std=c++11<br>
>nvcc -V $\geqslant$ 9.2
### windows
>[Visual Studio 2017 + cuda 9.2] or<br>
 [Visual Studio 2022 + cuda 11.4]

## Usage
 ### linux
 > cd travel_time_3d<br>
 > sh run.sh

### windows
>Create your own visual studio project, drag the code in and compile it.

## Example of running results (GPU:Tesla V100)
>sh run.sh <br>
>DEBUG MODEL<br>
>demo/model/cube200.gms<br>
><br>
>Grid_X_Num:  201; INTERVAL:     1; RANGE:[0, 200]<br>
>Grid_Y_Num:  201; INTERVAL:     1; RANGE:[0, 200]<br>
>Grid_Z_Num:  201; INTERVAL:     1; RANGE:[0, 200]<br>
>COST TIME1:0.26s<br>
>COST TIME2:0.26s<br>
>COST TIME3:0.29s<br>
>COST TIME4:0.32s<br>
>travel_time_3d_1rd:time_error_mean:  0.000837048<br>
>travel_time_3d_1rd:time_error_max :  0.0013247<br>
>travel_time_3d_1rd:time_error_min :  -2.32831e-10<br>
><br>
>travel_time_3d_1rd_diag:time_error_mean:  0.000703205<br>
>travel_time_3d_1rd_diag:time_error_max :  0.000979394<br>
>travel_time_3d_1rd_diag:time_error_min :  -2.05273e-09<br>
><br>
>travel_time_3d_2rd:time_error_mean:  0.000201762<br>
>travel_time_3d_2rd:time_error_max :  0.000310671<br>
>travel_time_3d_2rd:time_error_min :  -2.32831e-10<br>
><br>
>travel_time_3d_2rd_diag:time_error_mean:  0.000196653<br>
>travel_time_3d_2rd_diag:time_error_max :  0.000282375<br>
>travel_time_3d_2rd_diag:time_error_min :  -2.32831e-10<br>

## License
MIT 