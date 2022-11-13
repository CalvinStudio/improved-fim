#ifndef JARVIS_DEVICE_GMS_MEAT
#define JARVIS_DEVICE_GMS_MEAT
#include "jarvis_device_gms_bones.hpp"
namespace jarvis
{
	inline GmsReader::GmsReader(std::string path)
	{
		gms_model_path = path;
		read_gms_data();
	}
	inline void GmsReader::read_gms_data()
	{
		int SIZE_FILE_H = 240;
		int SIZE_DATA_H = 64;
		GmsFileHeader FileHeader;
		Field<GmsDataHeader> DataHeader;
		FILE *fp;
#ifdef __linux__
		fp = fopen(gms_model_path.c_str(), "rb");
#else
		errno_t err = fopen_s(&fp, gms_model_path.c_str(), "rb");
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
			grid.set_ndl(FileHeader.n_rows,
						 FileHeader.n_cols,
						 FileHeader.n_slices,
						 FileHeader.d_rows,
						 FileHeader.d_cols,
						 FileHeader.d_slices,
						 FileHeader.l_rows,
						 FileHeader.l_cols,
						 FileHeader.l_slices);
			para_num = FileHeader.para_num;
			DataHeader.alloc(para_num);
			all_data.alloc(para_num);
			for (int i = 0; i < para_num; i++)
			{
				all_data[i].alloc(grid.n_elem);
				fread(DataHeader.get_mem_p(), SIZE_DATA_H, 1, fp);
				fread(all_data(i).get_mem_p(), grid.n_elem * sizeof(float), 1, fp);
			}
			fclose(fp);
		}
		else
		{
			printf("read_gms_data():\033[41;37m[ERROR]:\033[0mFile open error!");
			std::abort();
		}
	}
	inline void GmsReader::read_gms_by_order_to_ffield(ffield &ffield_obj)
	{
		if (para_order < para_num)
		{
			ffield_obj.alloc(grid);
			for (int j = 0; j < grid.n_cols; j++)
				for (int i = 0; i < grid.n_rows; i++)
					for (int k = 0; k < grid.n_slices; k++)
						ffield_obj(i, j, k) = all_data[para_order][i * grid.n_slices + j * grid.n_rows * grid.n_slices + k];
			para_order++;
		}
		else
		{
			printf("ERROR:There are no more parameters to read!");
			std::abort();
		}
	}

	inline void GmsReader::read_gms_by_order_to_fcufield(fcufld &fcufld_obj)
	{
		if (para_order < para_num)
		{
			fcufld_obj.cu_alloc(MemType::npin, grid);
			for (int j = 0; j < grid.n_cols; j++)
				for (int i = 0; i < grid.n_rows; i++)
					for (int k = 0; k < grid.n_slices; k++)
						fcufld_obj(i, j, k) = all_data[para_order][i * grid.n_slices + j * grid.n_rows * grid.n_slices + k];
			fcufld_obj.cu_copy_h2d();
			cudaStreamSynchronize(jarvis_default_stream);
			para_order++;
		}
		else
		{
			printf("ERROR:There are no more parameters to read!");
			std::abort();
		}
	}

#ifdef JARVIS_DEBUG
	inline void GmsReader::print_info()
	{
		cout << "Grid Information:" << endl;
		cout << "para_order:" << para_order << ";"
			 << "para_num:" << para_num << ";" << endl;
		grid.print_info("GMS_MODEL:");
	}
#endif
}
#endif
