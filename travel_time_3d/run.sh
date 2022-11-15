#!/bin/bash
#*util
TARGET="travel_time_3d"
STD="-std=c++11"
NVCC="nvcc"
CC="g++"
OPTION="-O3 -w -g"

nvcc ${STD} -o ${TARGET} travel_time_3d_4_device.cu travel_time_3d_5_main.cpp ${OPTION}

./${TARGET}