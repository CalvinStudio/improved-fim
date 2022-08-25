#pragma once
#ifndef RACC_HOST_TMAT_FUNCTION
#define RACC_HOST_TMAT_FUNCTION
//**********************************Developer*************************************
// 2021.03.02 BY CAIWEI CALVIN CAI
//********************************************************************************
#include "racc_host_81_Tmat_meat.hpp"
//********************************CLASS_TEMPLATE**********************************
namespace racc
{
    template <typename T>
    Mat<T> min(const Mat<T> &a, const Mat<T> &b)
    {
#ifdef RACC_DEBUG
        _ASSERT_IS_REAL_NUMBER;
        if (a.n_rows == b.n_rows && a.n_cols == b.n_cols)
        {
#endif
            Mat<T> ab_min(a.n_rows, a.n_cols);
            for (int i = 0; i < a.n_rows; i++)
            {
                for (int j = 0; j < a.n_cols; j++)
                {
                    ab_min(i, j) = racc::min<T>(a(i, j), b(i, j));
                }
            }
            return ab_min;
#ifdef RACC_DEBUG
        }
        else
        {
            printf("min(Mat,Mat):[ERROR]:The two parameters are inconsistent!\n");
            RACC_ERROR_EXIT;
        }
#endif
    }

    inline void mat_save3c(Mat<fcomp3> _data, fcomp3::c _c, string _name, SaveFormat _save_format)
    {
        fmat _tmp(_data.n_rows, _data.n_cols);
        if (_c == fcomp3::c::x)
        {
            for (int i = 0; i < _data.n_rows; i++)
                for (int j = 0; j < _data.n_cols; j++)
                    _tmp(i, j) = _data(i, j).x;
            _name.append("_x");
        }
        else if (_c == fcomp3::c::y)
        {
            for (int i = 0; i < _data.n_rows; i++)
                for (int j = 0; j < _data.n_cols; j++)
                    _tmp(i, j) = _data(i, j).y;
            _name.append("_y");
        }
        else if (_c == fcomp3::c::z)
        {
            for (int i = 0; i < _data.n_rows; i++)
                for (int j = 0; j < _data.n_cols; j++)
                    _tmp(i, j) = _data(i, j).z;
            _name.append("_z");
        }
        _tmp.save(_name, _save_format);
    }
}
#endif