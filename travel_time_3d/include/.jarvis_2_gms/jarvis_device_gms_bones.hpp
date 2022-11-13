#ifndef JARVIS_DEVICE_GMS_BONES
#define JARVIS_DEVICE_GMS_BONES
#include ".jarvis_device_gms_header_in.h"
namespace jarvis
{
	struct GmsFileHeader
	{
		short int FileFormat;
		char ModelName[64];
		short int Year;
		char Mouth;
		char Day;
		char Hour;
		char Minutes;
		char Seconds;
		char type;
		short int para_num;
		float l_rows;
		float l_cols;
		float l_slices;
#ifdef __linux__
		int n_rows;
		int n_cols;
		int n_slices;
#endif
#ifdef _WIN32
		long n_rows;
		long n_cols;
		long n_slices;
#endif
		float d_rows;
		float d_cols;
		float d_slices;
		char Reserved[127];
	};
	struct GmsDataHeader
	{
		char dataName[64];
		char Reserved[64];
	};
	class GmsReader
	{
	public:
		Frame grid;
		Field<ffield> all_data;
		GmsReader(std::string path);
		void read_gms_by_order_to_ffield(ffield &field_obj);
		void read_gms_by_order_to_fcufield(fcufld &field_obj);
#ifdef JARVIS_DEBUG
		void print_info();
#endif
	private:
		int para_order = 0;
		int para_num;
		std::string gms_model_path;
		void read_gms_data();
	};
}
#endif
