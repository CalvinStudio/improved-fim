TRAVEL_CU=travel_time_3d_3_device.cu
TRAVEL_CPP=travel_time_3d_4_host.cpp
TRAVEL_MAIN=travel_time_3d_5_main.cpp
NVCC=nvcc
OPTION=-w -O2
EXE=main
STD=-std=c++11
travel:
	$(NVCC) $(STD) -g -o $(EXE) $(TRAVEL_MAIN) $(TRAVEL_CPP) $(TRAVEL_CU) $(OPTION)
