#!/bin/bash
#*util
TARGET="travel_time_3d"
STD="-std=c++11"
NVCC="nvcc"
CC="g++"
TEST_PROJ="demo"
OPTION="-O3 -w -g"

nvcc ${STD} -o ${TARGET} travel_time_3d_6_device.cu travel_time_3d_7_main.cpp ${OPTION}

./${TARGET} ${TEST_PROJ}