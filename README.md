![](jarvis_min.jpg =400x300)
# open_travel_time_3d_gpu

## Background
Traveltime calculation is widely used in seismic tomography,migration imaging, earthquake location, etc Fast Iterative Method(FIM) is one of the fastest methods for traveltime calculation since it can be easily accelerated by running in parallel, but its accuracy is poor and it cannot satisfy actual demands.Although the computational accuracy of FIM can be improved simply by increasing the grid density, this results in a rapid increase in computational workload and a significant reduction in computational efficiency. This study incorporates the high-order difference schemes, diagonal node calculation, and double-grid technique into the FIM to address this issue. The improved FIM can substantially increase the computational accuracy while essentially maintaining its original computational efficiency.

## **Key Points**
1. A novel FIM based on higher-order finite differences can considerably improve the accuracy of numerical solution of Eikonal equation.

2. Introducing diagonal node calculation and the double-grid technique into FIM can further improve calculation accuracy of the solution.

3. The improved FIM has a smaller increase in computational complexity, compared to the increase in accuracy.

## Software testing environment
### linux (Author's test environment:CentOS)
>g++ -std=c++11
>nvcc -V $\geqslant$ 9.2
### windows
>[Visual Studio 2017 + cuda 9.2] or [Visual Studio 2022 + cuda 11.4]

## Usage
 ### linux
 > cd travel_time_3d
 > sh run.sh

### windows
>Create your own visual studio project, drag the code in and compile it.
