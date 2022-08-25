#pragma once
#ifndef RACC_ENUM_H
#define RACC_ENUM_H
//***************************************Developer*************************************
// 2020.04.10 BY CAIWEI CALVIN CAI
//*************************************************************************************
#include "racc_host_01_const.hpp"
//*************************************STRUCR_DEFINE***********************************
namespace racc
{
  //****************************************ENUM***************************************
  enum class MemState
  {
    released = -1,
    inital = 0,
    allocated = 1
  };
  //
  enum class MemType
  {
    npin = 0,
    pin
  };
  //
  enum class ArrayType
  {
    null = 0,
    vec = 1,
    mat = 2,
    cube = 3,
    line = 10,
    mesh = 20,
    grid = 30
  };
  //
  enum class FrameType
  {
    null = 0,
    vec = 1,
    mat = 2,
    cube = 3,
    line = 10,
    mesh = 20,
    grid = 30
  };
  //
  enum class Fill
  {
    zeros,
    ones,
    randu = 23
  };
  //
  enum class SaveFormat
  {
    ascii_txt,
    ascii_grd,
    ascii_xyz,
    binary_raw,
    binary_fld
  };
  //
  enum class Position
  {
    top = 1,
    bottom,
    left,
    right,
    front,
    back
  };
  //
  enum class TransType
  {
    h2h = 0,
    h2d,
    d2h,
    d2d
  };
}
//*************************************************************************************
#endif