#pragma once
#ifndef RACC_HOST_SEIS_GMSREADER_H_MEAT
#define RACC_HOST_SEIS_GMSREADER_H_MEAT
#include "racc_host_seis_30_gms_bones.hpp"
namespace racc
{
	inline CGmsReader::CGmsReader(std::string path)
	{
		Path = path;
		ReadGMSModel();
	}

	inline void CGmsReader::ReadGMSModel()
	{
		int SIZE_FILE_H = 240;
		int SIZE_DATA_H = 64;
		GmsFileHeader FileHeader;
		Vec<GmsDataHeader> DataHeader;
		FILE *fp;
#ifdef __linux__
		fp = fopen(Path.c_str(), "rb");
#else
		errno_t err = fopen_s(&fp, Path.c_str(), "rb");
#endif
		if (
#ifdef __linux__
			fp
#else
			err == 0 && fp
#endif
		)
		{
			fread(&FileHeader, SIZE_FILE_H, 1, fp);
			grid.n_rows = FileHeader.nx;
			grid.n_cols = FileHeader.ny;
			grid.n_slices = FileHeader.nz;
			grid.l_rows = FileHeader.x0;
			grid.l_cols = FileHeader.y0;
			grid.l_slices = FileHeader.z0;
			grid.d_rows = FileHeader.dx;
			grid.d_cols = FileHeader.dy;
			grid.d_slices = FileHeader.dz;
			grid.r_rows = grid.l_rows + (double(grid.n_rows) - 1) * grid.d_rows;
			grid.r_cols = grid.l_cols + (double(grid.n_cols) - 1) * grid.d_cols;
			grid.r_slices = grid.l_slices + (double(grid.n_slices) - 1) * grid.d_slices;
			ParaNum = FileHeader.ParaNum;
			DataHeader.alloc(ParaNum);
			long GridPointSum = grid.n_rows * grid.n_cols * grid.n_slices;

			AllData.alloc(ParaNum, GridPointSum);

			for (int i = 0; i < ParaNum; i++)
			{
				fread((char *)&DataHeader.mem[i], SIZE_DATA_H, 1, fp);
				fread((char *)AllData.mem[i], sizeof(float) * GridPointSum, 1, fp);
			}
			fclose(fp);
		}
		else
		{
			printf("ReadGMSModel():[ERROR]:File open error!");
			RACC_ERROR_EXIT;
		}
	}
	template <typename T>
	inline Cube<T> CGmsReader::ReadGMSByOrderToCube()
	{
		fvec ParaData;
		long GridPointSum = grid.n_rows * grid.n_cols * grid.n_slices;
		if (ParaOrder < ParaNum)
		{
			ParaData.alloc(GridPointSum);
			ParaData.mem = AllData.mem[ParaOrder];
			Cube<T> ParaGridTmp(grid);
			for (int i = 0; i < grid.n_rows; i++)
				for (int j = 0; j < grid.n_cols; j++)
					for (int k = 0; k < grid.n_slices; k++)
						ParaGridTmp(i, j, k) = ParaData.p4gms(grid, i, j, k);

			ParaOrder++;
			return ParaGridTmp;
		}
		else
		{
			_PLOT_LINE;
			printf("ERROR:There are no more parameters to read!");
			RACC_ERROR_EXIT;
		}
	}
	template <typename T>
	inline void CGmsReader::ReadGMSByOrderToField(Field<T> &field_obj)
	{
		fvec ParaData;
		long GridPointSum = grid.n_rows * grid.n_cols * grid.n_slices;
		if (ParaOrder < ParaNum)
		{
			ParaData.alloc(GridPointSum);
			ParaData.mem = AllData.mem[ParaOrder];
			Frame f_t = ToFrame(grid);
			field_obj.alloc(f_t);
			for (int i = 0; i < grid.n_rows; i++)
				for (int j = 0; j < grid.n_cols; j++)
					for (int k = 0; k < grid.n_slices; k++)
						field_obj(i, j, k) = ParaData.p4gms(grid, i, j, k);
			ParaOrder++;
		}
		else
		{
			_PLOT_LINE;
			printf("ERROR:There are no more parameters to read!");
			RACC_ERROR_EXIT;
		}
	}

#ifdef RACC_USE_ARMA
	template <typename T>
	inline arma::Grid<T> CGmsReader::ReadGMSByOrderToArmaGrid()
	{
		fvec ParaData;
		long GridPointSum = grid.n_rows * grid.n_cols * grid.n_slices;
		if (ParaOrder < ParaNum)
		{
			ParaData.alloc(GridPointSum);
			ParaData.mem = AllData.mem[ParaOrder];
			arma::Grid<T> ParaSpaceTmp(grid);
			for (int i = 0; i < grid.n_rows; i++)
				for (int j = 0; j < grid.n_cols; j++)
					for (int k = 0; k < grid.n_slices; k++)
						ParaSpaceTmp(i, j, k) = ParaData.p4gms(grid, i, j, k);

			ParaOrder++;
			return ParaSpaceTmp;
		}
		else
		{
			_PLOT_LINE
			printf("ERROR:There are no more parameters to read!");
			RACC_ERROR_EXIT;
		}
	}
#endif
#ifdef RACC_DEBUG
	inline void CGmsReader::PrintInfo()
	{
		_PLOT_LINE;
		cout << "Grid Information:" << endl;
		cout << "ParaOrder:" << ParaOrder << ";"
			 << "ParaNum:" << ParaNum << ";" << endl;
		grid.PrintGridInfo("GMS_MODEL:");
	}
#endif

	inline CGmsReader2D::CGmsReader2D(string path)
	{
		Path = path;
	}

	inline void CGmsReader2D::ReadGMSModel2D()
	{
		int SIZE_FILE_H = 240;
		int SIZE_DATA_H = 64;
		GmsFileHeader FileHeader;
		Vec<GmsDataHeader> DataHeader;
		fvec ParaData;
		FILE *fp;
#ifdef __linux__
		fp = fopen(Path.c_str(), "rb");
#else
		errno_t err = fopen_s(&fp, Path.c_str(), "rb");
#endif
		if (
#ifdef __linux__
			fp
#else
			err == 0 && fp
#endif
		)
		{
			fread(&FileHeader, SIZE_FILE_H, 1, fp);

			mesh.n_rows = FileHeader.nx;
			mesh.n_cols = FileHeader.nz;
			mesh.l_rows = FileHeader.x0;
			mesh.l_cols = FileHeader.z0;
			mesh.d_rows = FileHeader.dx;
			mesh.d_cols = FileHeader.dz;
			mesh.r_rows = mesh.l_rows + (double(mesh.n_rows) - 1.0) * mesh.d_rows;
			mesh.r_cols = mesh.l_cols + (double(mesh.n_cols) - 1.0) * mesh.d_cols;

			ParaGrid.grid.n_rows = FileHeader.nx;
			ParaGrid.grid.n_cols = FileHeader.ny;
			ParaGrid.grid.n_slices = FileHeader.nz;

			ParaNum = FileHeader.ParaNum;
			DataHeader.alloc(ParaNum);
			long GridPointSum = ParaGrid.grid.n_rows * ParaGrid.grid.n_cols * ParaGrid.grid.n_slices;

			AllData.alloc(ParaNum, GridPointSum);

			for (int i = 0; i < ParaNum; i++)
			{
				fread((char *)&DataHeader.mem[i], SIZE_DATA_H, 1, fp);
				fread((char *)AllData.mem[i], sizeof(float) * GridPointSum, 1, fp);
			}
			fclose(fp);
		}
		else
		{
			printf("ReadGMSModel2D():[ERROR]:File open error!");
			RACC_ERROR_EXIT;
		}
	}
	inline fmat CGmsReader2D::ReadGMSByOrder()
	{
		fvec ParaData;
		long GridPointSum = ParaGrid.grid.n_rows * ParaGrid.grid.n_cols * ParaGrid.grid.n_slices;
		if (ParaOrder < ParaNum)
		{
			ParaData.alloc(GridPointSum);
			ParaData.mem = AllData.mem[ParaOrder];
			fmat ParaMesh;
			ParaMesh.alloc(mesh);
			for (int i = 0; i < mesh.n_rows; i++)
				for (int k = 0; k < mesh.n_cols; k++)
					ParaMesh(i, k) = ParaData.p4gms(ParaGrid.grid, i, int(ParaGrid.grid.n_cols / 2.0 + 1), k);
			ParaOrder++;
			return ParaMesh;
		}
		else
		{
			_PLOT_LINE;
			printf("ERROR:There are no more parameters to read!");
			RACC_ERROR_EXIT;
		}
	}

	inline ffield CGmsReader2D::ReadGMSByOrderToField()
	{
		fvec ParaData;
		long GridPointSum = ParaGrid.grid.n_rows * ParaGrid.grid.n_cols * ParaGrid.grid.n_slices;
		if (ParaOrder < ParaNum)
		{
			ParaData.alloc(GridPointSum);
			ParaData.mem = AllData.mem[ParaOrder];
			ffield ParaMesh;
			ParaMesh.alloc(ToFrame(mesh));
			for (int i = 0; i < mesh.n_rows; i++)
				for (int j = 0; j < mesh.n_cols; j++)
					ParaMesh(i, j) = ParaData.p4gms(ParaGrid.grid, i, int(ParaGrid.grid.n_cols / 2.0 + 1), j);
			ParaOrder++;
			return ParaMesh;
		}
		else
		{
			_PLOT_LINE;
			printf("ERROR:There are no more parameters to read!");
			RACC_ERROR_EXIT;
		}
	}

#ifdef RACC_DEBUG
	inline void CGmsReader2D::PrintInfo()
	{
		_PLOT_LINE;
		cout << "Grid Information:" << endl;
		cout << "ParaOrder:" << ParaOrder << ";"
			 << "ParaNum:" << ParaNum << ";";
	}
#endif
}
#endif
