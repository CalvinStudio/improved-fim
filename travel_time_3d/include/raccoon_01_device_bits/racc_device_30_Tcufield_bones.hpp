#ifndef RACC_DEVICE_TCUFIELD_BONES
#define RACC_DEVICE_TCUFIELD_BONES
//**********************************Developer*************************************
// 2021.05.14 BY CAIWEI CALVIN CAI
//********************************************************************************
#include "racc_device_21_fun_meat.hpp"
//********************************CLASS_TEMPLATE**********************************
namespace racc
{
	template <typename eT>
	class cuField : public Field<eT>
	{
	protected:
		MemState cu_mem_state = MemState::inital;
		MemType mem_type = MemType::npin;
		//
	public:
		eT *cu_mem;
		//
	public:
		cuField();
		cuField(string _name);
		cuField(eT *_mem_ptr);
		cuField(MemType _mem_type, uint32_t _n_rows, uint32_t _n_cols = 1, uint32_t _n_slices = 1);
		cuField(MemType _mem_type, const Frame &_frame);
		void cu_alloc(MemType _mem_type, uint32_t _n_rows, uint32_t _n_cols = 1, uint32_t _n_slices = 1);
		void cu_alloc(MemType _mem_type, const Frame &_frame);
		void cu_alloc_only_host(MemType _mem_type, uint32_t _n_rows, uint32_t _n_cols = 1, uint32_t _n_slices = 1);
		void cu_alloc_only_host(MemType _mem_type, const Frame &_frame);
		void cu_alloc_only_device(uint32_t _n_rows, uint32_t _n_cols = 1, uint32_t _n_slices = 1);
		void cu_alloc_only_device(const Frame &_frame);
		void cu_copy_h2d();
		void cu_copy_d2h();
		void cu_copy_h2d(cuField<eT> &_dst);
		void cu_copy_d2h(cuField<eT> &_dst);
		void cu_stream_copy_h2d(cudaStream_t s);
		void cu_stream_copy_d2h(cudaStream_t s);
		void cu_stream_copy_h2d(cuField<eT> &_dst, cudaStream_t s);
		void cu_stream_copy_d2h(cuField<eT> &_dst, cudaStream_t s);
		void cu_save(string file_path, SaveFormat _format);
		void cu_set_zero();
		MemState cu_get_mem_state();
		//
		void ToCudaField(MemType _mem_type, const Field<eT> &obj);
		//
		void del();
		void host_del();
		void device_del();
		eT cu_get_value(uint32_t i, uint32_t j = 0, uint32_t k = 0);
#ifdef RACC_DEBUG
	public:
#ifdef RACC_MEMORY_RELEASE_MANAGEMENT
		~cuField();
#endif
#endif
	};
	//******************************************************************
	typedef cuField<double> cufld;
	typedef cuField<double> dcufld;
	typedef cuField<float> fcufld;
	typedef cuField<int> icufld;
	//******************************************************************
}
#endif
