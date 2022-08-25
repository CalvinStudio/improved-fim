#pragma once
#ifndef RACC_HOST_TCUBE_FUNCTION
#define RACC_HOST_TCUBE_FUNCTION
//**********************************Developer*************************************
// 2021.03.02 BY CAIWEI CALVIN CAI
//********************************************************************************
#include "racc_host_91_Tcube_meat.hpp"
//********************************CLASS_TEMPLATE**********************************
namespace racc
{
    template <typename T>
    Cube<T> min(const Cube<T> &a, const Cube<T> &b)
    {
#ifdef RACC_DEBUG
        _ASSERT_IS_REAL_NUMBER;
        if (a.n_rows == b.n_rows && a.n_cols == b.n_cols && a.n_slices == b.n_slices)
        {
#endif
            Cube<T> ab_min(a.n_rows, a.n_cols, a.n_slices);
            for (int i = 0; i < a.n_rows; i++)
            {
                for (int j = 0; j < a.n_cols; j++)
                {
                    for (int k = 0; k < a.n_slices; k++)
                    {
                        ab_min(i, j, k) = racc::min<T>(a(i, j, k), b(i, j, k));
                    }
                }
            }
            return ab_min;
#ifdef RACC_DEBUG
        }
        else
        {
            printf("min(Cube,Cube):[ERROR]:The two parameters are inconsistent!\n");
            RACC_ERROR_EXIT;
        }
#endif
    }

    template <typename T>
    inline void cube_save3c(Cube<T> _data, fcomp3::c _c, string _name, SaveFormat _save_format)
    {
        fcube _tmp(_data.n_rows, _data.n_cols, _data.n_slices);
        if (_c == fcomp3::c::x)
        {
            for (int i = 0; i < _data.n_rows; i++)
                for (int j = 0; j < _data.n_cols; j++)
                    for (int k = 0; k < _data.n_slices; k++)
                        _tmp(i, j, k) = _data(i, j, k).x;
            _name.append("_x");
        }
        else if (_c == fcomp3::c::y)
        {
            for (int i = 0; i < _data.n_rows; i++)
                for (int j = 0; j < _data.n_cols; j++)
                    for (int k = 0; k < _data.n_slices; k++)
                        _tmp(i, j, k) = _data(i, j, k).y;
            _name.append("_y");
        }
        else if (_c == fcomp3::c::z)
        {
            for (int i = 0; i < _data.n_rows; i++)
                for (int j = 0; j < _data.n_cols; j++)
                    for (int k = 0; k < _data.n_slices; k++)
                        _tmp(i, j, k) = _data(i, j, k).z;
            _name.append("_z");
        }
        _tmp.save(_name, _save_format);
    }

}
#endif