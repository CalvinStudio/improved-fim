#pragma once
#ifndef RACC_HOST_H_IN
#define RACC_HOST_H_IN
//**********************************RACC***********************************************
//2020.04.10 BY CAIWEI CALVIN CAI
//*************************************************************************************
// #include "../../_racc_public/public_header_out.h"
#include <array>
#include <assert.h>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>
#include <typeinfo>
#include <vector>
#ifdef RACC_USE_ARMA
#include "../armadillo_racc_cpu_bits/_arma_racc_cpu_header_out.h"
#endif
using namespace std;
#endif