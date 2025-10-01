#ifndef INCLUDE_CUDA_PSD_EST_PROC_H
#define INCLUDE_CUDA_PSD_EST_PROC_H

//--------------------------------------------------------------------------------------------------------------------------

#include <string>
#include <cstdbool>
#include <cstdint>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class cuda_psd_est_proc
{
public: //==================================================================================================================

    // constructor(s) / destructor
    cuda_psd_est_proc( uint32_t gpu, uint32_t num_rows, uint32_t fft_size );
   ~cuda_psd_est_proc( void );

    // public methods
    bool compute( cuFloatComplex *p_signal, uint32_t signal_len, float *p_psd_est_buffer, uint32_t psd_est_buffer_len );

private: //=================================================================================================================

    // private constants
    const std::string NAME = std::string("PSD_EST_PROC");

    // private variables
    uint32_t this_gpu;
    uint32_t this_fft_size;
    uint32_t this_num_rows;
    uint32_t this_num_items;
    uint32_t this_complex_row_len;
    uint32_t this_float_row_len;
    uint32_t this_signal_len;
    uint32_t this_window_len;
    uint8_t *this_p_data_gpu;
    uint8_t *this_p_window_gpu;
    uint8_t *this_p_float_row_gpu;

    cudaStream_t       this_stream;
    cufftHandle        this_fft_handle;
    std::vector<float> this_zeros_row;

    // private methods
    bool zero_float_row( float *p_row_gpu );
    bool generate_window( void );
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_CUDA_PSD_EST_PROC_H
