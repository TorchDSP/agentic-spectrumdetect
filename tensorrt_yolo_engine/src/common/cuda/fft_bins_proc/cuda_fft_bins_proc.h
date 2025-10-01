#ifndef INCLUDE_CUDA_FFT_BINS_PROC_H
#define INCLUDE_CUDA_FFT_BINS_PROC_H

//--------------------------------------------------------------------------------------------------------------------------

#include <string>
#include <cstdbool>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class cuda_fft_bins_proc
{
public: //==================================================================================================================

    // constructor(s) / destructor
    cuda_fft_bins_proc( uint32_t gpu, uint32_t num_fft_bins );
   ~cuda_fft_bins_proc( void );

    // public methods
    float compute_stderr_mean_diff( float *p_fft_bins_0, float *p_fft_bins_1, cudaStream_t stream );

private: //=================================================================================================================

    // private constants
    const std::string NAME = std::string("FFT_BINS_PROC");

    // private variables
    uint32_t this_gpu;
    uint32_t this_num_fft_bins;
    uint32_t this_fft_bins_len;
    uint32_t this_num_threads_per_block_bits;
    float   *this_p_fft_bins_0_gpu;
    float   *this_p_fft_bins_1_gpu;
    float   *this_p_pow_diff_0_gpu;
    float   *this_p_pow_diff_1_gpu;
    float   *this_p_pow_diff_sqrd_0_gpu;
    float   *this_p_pow_diff_sqrd_1_gpu;
    float   *this_p_stderr_mean_diff_gpu;
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_CUDA_FFT_BINS_PROC_H
