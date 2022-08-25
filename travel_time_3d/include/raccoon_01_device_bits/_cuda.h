#pragma once
#ifndef RACC_CUDA_HEADER
#define RACC_CUDA_HEADER
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#ifndef _PUBLIC_MACRO_H
#define _PUBLIC_MACRO_H

#ifdef __linux__
#define nullptr (0)
#endif

#ifndef FILE_NAME_SIZE_MAX
#define FILE_NAME_SIZE_MAX (256)
#endif
#ifndef UINT_MAX
#define UINT_MAX (0x7fffffff)
#endif
#ifndef INT_MAX
#define INT_MAX (2147483647)
#endif
// draw console split line
#define _PLOT_LINE printf("-----------------------------------------------------------------------------------|\n")

#define _PLOT_ERROR_LINE printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxERRORxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|\n");
// END draw console split line

#define _COUT(a) \
    std::cout << "[" << #a << "]: " << a << std::endl

//* time
#define TIC(s)                          \
    clock_t clockBegin##s, clockEnd##s; \
    clockBegin##s = clock();

#define TOC(s, string)                                                               \
    clockEnd##s = clock();                                                           \
    double elapsed_time##s = (double)(clockEnd##s - clockBegin##s) / CLOCKS_PER_SEC; \
    cout << string << elapsed_time##s << "s" << endl;

#define OMP_TIC(s) \
    double start##s = omp_get_wtime();

#define OMP_TOC(s, _string)          \
    double end##s = omp_get_wtime(); \
    cout << _string << (double)(end##s - start##s) << "s" << endl;

#define MPI_TIC(s) \
    double mpi_start##s = MPI_Wtime();

#define MPI_TOC(s, mpi_rank, _string) \
    double mpi_end##s = MPI_Wtime();  \
    if (mpi_rank == 0)                \
        cout << _string << (double)(mpi_end##s - mpi_start##s) << "s" << endl;

//* for debug
#define RACC_ERROR_EXIT \
    char a = getchar(); \
    throw

#define _ASSERT_IS_ARMA_NUMBER                                        \
    static_assert(std::is_same<T, double>::value ||                   \
                      std::is_same<T, float>::value ||                \
                      std::is_same<T, int>::value ||                  \
                      std::is_same<T, unsigned int>::value ||         \
                      std::is_same<T, std::complex<double>>::value || \
                      std::is_same<T, std::complex<float>>::value,    \
                  "Template type is not ARMA numeric type!");

#define _ASSERT_IS_REAL_NUMBER                                      \
    static_assert(std::is_same<T, long double>::value ||            \
                      std::is_same<T, double>::value ||             \
                      std::is_same<T, float>::value ||              \
                      std::is_same<T, long int>::value ||           \
                      std::is_same<T, unsigned long int>::value ||  \
                      std::is_same<T, short int>::value ||          \
                      std::is_same<T, unsigned short int>::value || \
                      std::is_same<T, int>::value ||                \
                      std::is_same<T, unsigned int>::value,         \
                  "Template type is not REAL numeric type!");

#define IS_NUMBER(a)                            \
    (string(typeid(a).name()) == "int" ||       \
     string(typeid(a).name()) == "short" ||     \
     string(typeid(a).name()) == "long" ||      \
     string(typeid(a).name()) == "long long" || \
     string(typeid(a).name()) == "float" ||     \
     string(typeid(a).name()) == "double" ||    \
     string(typeid(a).name()) == "long double")

#define IS_INT_NUMBER(a)                    \
    (string(typeid(a).name()) == "int" ||   \
     string(typeid(a).name()) == "short" || \
     string(typeid(a).name()) == "long" ||  \
     string(typeid(a).name()) == "long long")

#define IS_FLOAT_NUMBER(a)                   \
    (string(typeid(a).name()) == "float" ||  \
     string(typeid(a).name()) == "double" || \
     string(typeid(a).name()) == "long double")
// END for debug

#endif // !PUBLIC_MACRO_H
#endif