//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include <stdio.h>
#include <stdexcept>
#include "cuda_fft_bins_proc.h"

//-- constructor(s) --------------------------------------------------------------------------------------------------------

cuda_fft_bins_proc::cuda_fft_bins_proc( uint32_t gpu, uint32_t num_fft_bins )
{
    cudaError_t status = cudaSuccess;

    // initialize
    this_gpu                        = gpu;
    this_num_fft_bins               = num_fft_bins;
    this_fft_bins_len               = (uint32_t)(num_fft_bins * sizeof(float));
    this_num_threads_per_block_bits = 8;
    this_p_fft_bins_0_gpu           = nullptr;
    this_p_fft_bins_1_gpu           = nullptr;
    this_p_pow_diff_0_gpu           = nullptr;
    this_p_pow_diff_1_gpu           = nullptr;
    this_p_pow_diff_sqrd_0_gpu      = nullptr;
    this_p_pow_diff_sqrd_1_gpu      = nullptr;
    this_p_stderr_mean_diff_gpu     = nullptr;

    // allocate GPU buffers
    cudaSetDevice(gpu);

    status = cudaMalloc(&this_p_fft_bins_0_gpu, this_fft_bins_len);
    if ( status != cudaSuccess ) { goto FAILED; }

    status = cudaMalloc(&this_p_fft_bins_1_gpu, this_fft_bins_len);
    if ( status != cudaSuccess ) { goto FAILED_FREE_FFT_BINS_0; }

    status = cudaMalloc(&this_p_pow_diff_0_gpu, this_fft_bins_len);
    if ( status != cudaSuccess ) { goto FAILED_FREE_FFT_BINS_1; }

    status = cudaMalloc(&this_p_pow_diff_1_gpu, this_fft_bins_len);
    if ( status != cudaSuccess ) { goto FAILED_FREE_POW_DIFF_0; }

    status = cudaMalloc(&this_p_pow_diff_sqrd_0_gpu, this_fft_bins_len);
    if ( status != cudaSuccess ) { goto FAILED_FREE_POW_DIFF_1; }

    status = cudaMalloc(&this_p_pow_diff_sqrd_1_gpu, this_fft_bins_len);
    if ( status != cudaSuccess ) { goto FAILED_FREE_POW_DIFF_SQRD_0; }

    status = cudaMalloc(&this_p_stderr_mean_diff_gpu, sizeof(float));
    if ( status != cudaSuccess ) { goto FAILED_FREE_POW_DIFF_SQRD_1; }

    return;

FAILED_FREE_POW_DIFF_SQRD_1:
    cudaFree(this_p_pow_diff_sqrd_1_gpu);

FAILED_FREE_POW_DIFF_SQRD_0:
    cudaFree(this_p_pow_diff_sqrd_0_gpu);

FAILED_FREE_POW_DIFF_1:
    cudaFree(this_p_pow_diff_1_gpu);

FAILED_FREE_POW_DIFF_0:
    cudaFree(this_p_pow_diff_0_gpu);

FAILED_FREE_FFT_BINS_1:
    cudaFree(this_p_fft_bins_1_gpu);

FAILED_FREE_FFT_BINS_0:
    cudaFree(this_p_fft_bins_0_gpu);

FAILED:
    throw std::runtime_error(cuda_fft_bins_proc::NAME + " [EXCEPTION] ALLOCATE GPU MEMORY");
}

//-- destructor ------------------------------------------------------------------------------------------------------------

cuda_fft_bins_proc::~cuda_fft_bins_proc( void )
{
    // clean up
    cudaSetDevice(this_gpu);

    cudaFree(this_p_fft_bins_0_gpu);
    cudaFree(this_p_fft_bins_1_gpu);
    cudaFree(this_p_pow_diff_0_gpu);
    cudaFree(this_p_pow_diff_1_gpu);
    cudaFree(this_p_pow_diff_sqrd_0_gpu);
    cudaFree(this_p_pow_diff_sqrd_1_gpu);
    cudaFree(this_p_stderr_mean_diff_gpu);

    return;
}

//-- CUDA kernels ----------------------------------------------------------------------------------------------------------

// calculate the power difference between 2 sets of fft bins
__global__
void cudak_calc_pow_diff( float *p_pow_diff_out, float *p_pow_diff_sqrd_out, float *p_fft_bins_0_in,
                          float *p_fft_bins_1_in )
{
    int32_t idx      = ((blockIdx.x * blockDim.x) + threadIdx.x);
    float   pow_diff = (p_fft_bins_0_in[idx] - p_fft_bins_1_in[idx]);

    p_pow_diff_out[idx]      =  pow_diff;
    p_pow_diff_sqrd_out[idx] = (pow_diff * pow_diff);

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// sum via reduction [https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf]
__global__
void cudak_sum_reduce( float *p_out, float *p_in )
{
    extern __shared__ float shmem[];

    int32_t tid = threadIdx.x;

#if true
    int32_t idx = ((blockIdx.x * blockDim.x) + threadIdx.x);
    shmem[tid]  = p_in[idx];
#else
    int32_t idx = ((blockIdx.x * (blockDim.x << 1)) + threadIdx.x);
    shmem[tid]  = (p_in[idx] + p_in[idx + blockDim.x]);
#endif

    __syncthreads();

    for ( uint32_t s = (blockDim.x >> 1); s > 0; s >>= 1 )
    {
        if ( tid < s ) { shmem[tid] += shmem[tid + s]; }
        __syncthreads();
    }

    if ( tid == 0 ) { p_out[blockIdx.x] = shmem[0]; }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// compute the standard-error mean difference
__global__
void cudak_compute_stderr_mean_diff( float *p_stderr_mean_diff_out, float *p_pow_diff_in, float *p_pow_diff_sqrd_in,
                                     uint32_t num_thread_blocks_to_sum, uint32_t num_fft_bins )
{
    float pow_diff_sum      = 0.0f;
    float pow_diff_sqrd_sum = 0.0f;

    for ( uint32_t i = 0; i < num_thread_blocks_to_sum; i++ )
    {
        pow_diff_sum      += p_pow_diff_in[i];
        pow_diff_sqrd_sum += p_pow_diff_sqrd_in[i];
    }

   *p_stderr_mean_diff_out = sqrtf((num_fft_bins * pow_diff_sqrd_sum) - (pow_diff_sum * pow_diff_sum)) / num_fft_bins;

    return;
}

//-- public methods --------------------------------------------------------------------------------------------------------

// compute standard-error mean difference between two sets of fft bins
float cuda_fft_bins_proc::compute_stderr_mean_diff( float *p_fft_bins_0, float *p_fft_bins_1, cudaStream_t stream )
{
    dim3        num_thread_blocks     = (this_num_fft_bins >> this_num_threads_per_block_bits);
    dim3        num_threads_per_block = (                1 << this_num_threads_per_block_bits);
    float       stderr_mean_diff      = 0.0f;
    cudaError_t status                = cudaSuccess;

    cudaSetDevice(this_gpu);

    // transfer fft bins to the gpu
    status = cudaMemcpyAsync(this_p_fft_bins_0_gpu, p_fft_bins_0, this_fft_bins_len, cudaMemcpyHostToDevice, stream);
    if ( status != cudaSuccess ) { goto FAILED; }

    status = cudaMemcpyAsync(this_p_fft_bins_1_gpu, p_fft_bins_1, this_fft_bins_len, cudaMemcpyHostToDevice, stream);
    if ( status != cudaSuccess ) { goto FAILED; }

    // calculate the power difference between the 2 sets of fft bins
    cudak_calc_pow_diff<<< num_thread_blocks, num_threads_per_block, 0, stream >>>
      (this_p_pow_diff_0_gpu, this_p_pow_diff_sqrd_0_gpu, this_p_fft_bins_0_gpu, this_p_fft_bins_1_gpu);

    {
        // sum power difference values via reduction
        float   *p_pow_diff_in_gpu        = nullptr;
        float   *p_pow_diff_out_gpu       = nullptr;
        float   *p_pow_diff_sqrd_in_gpu   = nullptr;
        float   *p_pow_diff_sqrd_out_gpu  = nullptr;
        uint32_t shmem_size               = (uint32_t)(num_threads_per_block.x * sizeof(float));
        uint32_t num_thread_blocks_to_sum = this_num_fft_bins;
        uint32_t loop_cnt                 = 0;

        while ( true )
        {
            if ( (loop_cnt & 1) == 0 )
            {
                p_pow_diff_in_gpu       = this_p_pow_diff_0_gpu;
                p_pow_diff_out_gpu      = this_p_pow_diff_1_gpu;
                p_pow_diff_sqrd_in_gpu  = this_p_pow_diff_sqrd_0_gpu;
                p_pow_diff_sqrd_out_gpu = this_p_pow_diff_sqrd_1_gpu;
            }
            else
            {
                p_pow_diff_in_gpu       = this_p_pow_diff_1_gpu;
                p_pow_diff_out_gpu      = this_p_pow_diff_0_gpu;
                p_pow_diff_sqrd_in_gpu  = this_p_pow_diff_sqrd_1_gpu;
                p_pow_diff_sqrd_out_gpu = this_p_pow_diff_sqrd_0_gpu;
            }

            loop_cnt++;

            cudak_sum_reduce<<< num_thread_blocks, num_threads_per_block, shmem_size, stream >>>
              (p_pow_diff_out_gpu, p_pow_diff_in_gpu);

            cudak_sum_reduce<<< num_thread_blocks, num_threads_per_block, shmem_size, stream >>>
              (p_pow_diff_sqrd_out_gpu, p_pow_diff_sqrd_in_gpu);

            num_thread_blocks_to_sum = num_thread_blocks.x;

            if ( num_thread_blocks.x < (1 << this_num_threads_per_block_bits) ) { break; }
            num_thread_blocks.x >>= this_num_threads_per_block_bits;
        }

        // compute standard-error mean difference
        num_thread_blocks     = dim3(1);
        num_threads_per_block = dim3(1);

        cudak_compute_stderr_mean_diff<<< num_thread_blocks, num_threads_per_block, 0, stream >>>
          (this_p_stderr_mean_diff_gpu, p_pow_diff_out_gpu, p_pow_diff_sqrd_out_gpu,
           num_thread_blocks_to_sum, this_num_fft_bins);

        // transfer standard-error mean difference to the host
        status = cudaMemcpyAsync(&stderr_mean_diff, this_p_stderr_mean_diff_gpu, sizeof(stderr_mean_diff),
                                 cudaMemcpyDeviceToHost, stream);
        if ( status != cudaSuccess ) { goto FAILED; }

        cudaStreamSynchronize(stream);
    }

    return ( stderr_mean_diff );

FAILED:
    throw std::runtime_error(cuda_fft_bins_proc::NAME + " [EXCEPTION] COMPUTE STDERR MEAN DIFF");
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
