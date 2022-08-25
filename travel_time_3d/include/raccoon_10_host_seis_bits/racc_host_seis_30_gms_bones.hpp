#pragma once
#ifndef RACC_HOST_SEIS_GMSREADER_H_BONES
#define RACC_HOST_SEIS_GMSREADER_H_BONES
#include "racc_host_seis_11_fun_meat.hpp"
namespace racc
{
	struct GmsFileHeader
	{
		short int FileFormat; // 数据类型号
		char ModelName[64];	  // 模型名
		short int Year;		  // 模型生成时间，年
		char Mouth;			  // 模型生成时间，月
		char Day;			  // 模型生成时间，日
		char Hour;			  // 模型生成时间，时
		char Minutes;		  // 模型生成时间，分
		char Seconds;		  // 模型生成时间，秒
		char type;			  // 1 = 弹性
		short int ParaNum;	  // 参数个数
		float x0;			  // x起点坐标
		float y0;			  // y起点坐标
		float z0;			  // z起点坐标
#ifdef __linux__
		int nx; // x网格数
		int ny; // y网格数
		int nz; // z网格数
#endif
#ifdef _WIN32
		long nx; // x网格数
		long ny; // y网格数
		long nz; // z网格数
#endif
		float dx;			// x网格间距
		float dy;			// y网格间距
		float dz;			// z网格间距
		char Reserved[127]; // 保留空间
	};

	struct GmsDataHeader
	{
		char dataName[64]; // 参数数据名
		char Reserved[64]; // 保留空间
	};

	class CGmsReader2D
	{
	private:
		int ParaOrder = 0;
		int ParaNum;
		string Path;
		void ReadGMSModel2D();

	public:
		// fmat ParaMesh;
		fcube ParaGrid;
		fmat AllData;
		MeshFrame mesh;

	public:
		CGmsReader2D(string path);
		fmat ReadGMSByOrder();
		ffield ReadGMSByOrderToField();
#ifdef RACC_DEBUG
		void PrintInfo();
#endif
	};

	class CGmsReader
	{
	private:
		int ParaOrder = 0;
		int ParaNum;
		std::string Path;
		void ReadGMSModel();

	public:
		fmat AllData;
		GridFrame grid;

	public:
		CGmsReader(std::string path);
		template <typename T = float>
		Cube<T> ReadGMSByOrderToCube();
		template <typename T = float>
		void ReadGMSByOrderToField(Field<T> &field_obj);
#ifdef RACC_USE_ARMA
		template <typename T>
		arma::Grid<T> ReadGMSByOrderToArmaGrid();
#endif

#ifdef RACC_DEBUG
		void PrintInfo();
#endif
	};
}
#endif
