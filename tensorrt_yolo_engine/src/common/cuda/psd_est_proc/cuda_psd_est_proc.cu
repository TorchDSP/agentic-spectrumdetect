//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include <stdio.h>
#include <stdexcept>
#include <float.h>

#include "cuda_psd_est_proc.h"

//-- constructor(s) --------------------------------------------------------------------------------------------------------

cuda_psd_est_proc::cuda_psd_est_proc( uint32_t gpu, uint32_t num_rows, uint32_t fft_size )
{
    std::string error_str   = std::string("");
    cudaError_t cuda_status = cudaSuccess;
    cufftResult fft_status  = CUFFT_SUCCESS;
    bool        ok          = false;

    // initialize
    this_gpu             = gpu;
    this_num_rows        = num_rows;
    this_fft_size        = fft_size;
    this_num_items       = (uint32_t)(this_num_rows  * this_fft_size);
    this_complex_row_len = (uint32_t)(this_fft_size  * sizeof(cuFloatComplex));
    this_float_row_len   = (uint32_t)(this_fft_size  * sizeof(float));
    this_signal_len      = (uint32_t)(this_num_items * sizeof(cuFloatComplex));
    this_window_len      = (uint32_t)(this_fft_size  * sizeof(float));
    this_p_data_gpu      = nullptr;
    this_p_window_gpu    = nullptr;
    this_p_float_row_gpu = nullptr;
    this_stream          = {};
    this_fft_handle      = {};

    cudaSetDevice(this_gpu);

    // allocate GPU buffers
    cuda_status = cudaMalloc(&this_p_data_gpu, this_signal_len);
    if ( cuda_status != cudaSuccess )
    {
        error_str = std::string("ALLOCATE SIGNAL GPU BUFFER");
        goto FAILED;
    }

    cuda_status = cudaMalloc(&this_p_window_gpu, this_window_len);
    if ( cuda_status != cudaSuccess )
    {
        error_str = std::string("ALLOCATE WINDOW GPU BUFFER");
        goto FAILED;
    }

    cuda_status = cudaMalloc(&this_p_float_row_gpu, this_float_row_len);
    if ( cuda_status != cudaSuccess )
    {
        error_str = std::string("ALLOCATE FLOAT ROW GPU BUFFER");
        goto FAILED;
    }

    // generate the window function
    ok = this->generate_window();
    if ( ! ok )
    {
        error_str = std::string("GENERATE WINDOW FUNCTION");
        goto FAILED;
    }

    // create a stream
    cuda_status = cudaStreamCreate(&this_stream);
    if ( cuda_status != cudaSuccess )
    {
        error_str = std::string("CREATE STREAM");
        goto FAILED;
    }

    // create the fft plan
    fft_status = cufftPlan1d(&this_fft_handle, this_fft_size, CUFFT_C2C, this_num_rows/*batches*/);
    if ( fft_status != CUFFT_SUCCESS )
    {
        error_str = std::string("CREATE FFT PLAN");
        goto FAILED_DESTROY_STREAM;
    }

    fft_status = cufftSetStream(this_fft_handle, this_stream);
    if ( fft_status != CUFFT_SUCCESS )
    {
        error_str = std::string("SET FFT STREAM");
        goto FAILED_DESTROY_FFT_PLAN;
    }

    // generate a row of zeros
    this_zeros_row.clear();
    for ( uint32_t i = 0; i < this_fft_size; i++ ) { this_zeros_row.push_back(0.0f); }

    return;

FAILED_DESTROY_FFT_PLAN:
    cufftDestroy(this_fft_handle);

FAILED_DESTROY_STREAM:
    cudaStreamDestroy(this_stream);

FAILED:
    // clean up
    if ( this_p_data_gpu      != nullptr ) { cudaFree(this_p_data_gpu      ); }
    if ( this_p_window_gpu    != nullptr ) { cudaFree(this_p_window_gpu    ); }
    if ( this_p_float_row_gpu != nullptr ) { cudaFree(this_p_float_row_gpu ); }

    throw std::runtime_error(cuda_psd_est_proc::NAME + " [EXCEPTION] " + error_str);

    return;
}

//-- destructor ------------------------------------------------------------------------------------------------------------

cuda_psd_est_proc::~cuda_psd_est_proc( void )
{
    // clean up
    cudaSetDevice(this_gpu);

    cudaFree(this_p_data_gpu);
    cudaFree(this_p_window_gpu);
    cudaFree(this_p_float_row_gpu);

    cufftDestroy(this_fft_handle);
    cudaStreamDestroy(this_stream);

    return;
}

//-- CUDA kernels ----------------------------------------------------------------------------------------------------------

// apply a window function to the signal
__global__
void cudak_apply_window( cuFloatComplex *p_signal_out, cuFloatComplex *p_signal_in, float *p_window_in,
                         uint32_t signal_window_len )
{
    int32_t idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if ( idx > signal_window_len ) { return; }

    cuFloatComplex *p_out       =  (p_signal_out+ idx);
    float           signal_real =  (p_signal_in + idx)->x;
    float           signal_imag =  (p_signal_in + idx)->y;
    float           window_val  = *(p_window_in + idx);

    p_out->x = (signal_real * window_val);
    p_out->y = (signal_imag * window_val);

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// execute an FFT shift
__global__
void cudak_fft_shift( cuFloatComplex *p_fft_in_out, uint32_t fft_size )
{
    int32_t idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if ( idx > fft_size ) { return; }

    uint32_t        half_fft_size = (uint32_t)(fft_size >> 1);
    cuFloatComplex *p_complex_1   = nullptr;
    cuFloatComplex *p_complex_2   = nullptr;
    cuFloatComplex  complex_tmp   = {/*real*/0.0f, /*imag*/0.0f};

    if ( idx < half_fft_size )
    {
        p_complex_1 = (cuFloatComplex *)(p_fft_in_out + idx);
        p_complex_2 = (cuFloatComplex *)(p_complex_1 + half_fft_size);

        complex_tmp.x  = p_complex_1->x;
        complex_tmp.y  = p_complex_1->y;
        p_complex_1->x = p_complex_2->x;
        p_complex_1->y = p_complex_2->y;
        p_complex_2->x = complex_tmp.x;
        p_complex_2->y = complex_tmp.y;
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// compute complex absolute value
__global__
void cudak_abs( float *p_data_out, cuFloatComplex *p_data_in, uint32_t data_len )
{
    int32_t idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if ( idx > data_len ) { return; }

    cuFloatComplex *p_in = (p_data_in + idx);
    float           real =  p_in->x;
    float           imag =  p_in->y;

   *(p_data_out + idx) = ((real * real) + (imag * imag));

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// find maximum value via reduction
__global__
void cudak_max_reduce( float *p_data_out, float *p_data_in, uint32_t data_len )
{
    extern __shared__ float shmem[];

    int32_t idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if ( idx > data_len ) { return; }

    int32_t tid     = threadIdx.x;
    float   max_val = FLT_MIN;

    shmem[tid] = p_data_in[idx];
    __syncthreads();

    if ( tid == 0 )
    {
        for ( uint32_t val = 0; val < blockDim.x; val++ ) { max_val = fmaxf(max_val, shmem[val]); }
       *(p_data_out + blockIdx.x) = max_val;
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// normalize based on a maximum value
__global__
void cudak_norm_max( float *p_data_out, float *p_data_in, uint32_t data_len, float *p_max_val )
{
    int32_t idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if ( idx > data_len ) { return; }

    float val          = *(p_data_in + idx);
   *(p_data_out + idx) =  (val / *p_max_val);

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// compute nlog10
__global__
void cudak_nlog10( float *p_data_out, float *p_data_in, uint32_t data_len, uint32_t n )
{
    int32_t idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if ( idx > data_len ) { return; }

    float value        = *(p_data_in + idx);
   *(p_data_out + idx) =  (n * log10f(value));

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// sum rows
__global__
void cudak_sum_rows( float *p_row_out, float *p_row_1_in, float *p_row_2_in, uint32_t row_len )
{
    int32_t idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if ( idx > row_len ) { return; }

    float value_1 = *(p_row_1_in + idx);
    float value_2 = *(p_row_2_in + idx);

   *(p_row_out + idx) = (value_1 + value_2);

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// average
__global__
void cudak_average( float *p_data_out, float *p_data_in, uint32_t data_len )
{
    int32_t idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if ( idx > data_len ) { return; }

    float value        = *(p_data_in + idx);
   *(p_data_out + idx) =  (value / (float)data_len);

    return;
}

//-- public methods --------------------------------------------------------------------------------------------------------

// compute PSD estimates
bool cuda_psd_est_proc::compute( cuFloatComplex *p_signal, uint32_t signal_len, float *p_psd_est_buffer,
                                 uint32_t psd_est_buffer_len )
{
    uint8_t    *p_data_gpu                 = nullptr;
    uint32_t    num_threads_per_block_bits = 8;
    dim3        num_thread_blocks          = dim3(this_fft_size >> num_threads_per_block_bits);
    dim3        num_threads_per_block      = dim3(            1 << num_threads_per_block_bits);
    uint32_t    maxred_shmem_size          = (uint32_t)(num_threads_per_block.x * sizeof(float));
    cufftResult fft_status                 = CUFFT_SUCCESS;
    cudaError_t cuda_status                = cudaSuccess;
    bool        ok                         = false;

    if ( signal_len         != this_signal_len    ) { goto FAILED; }
    if ( psd_est_buffer_len != this_float_row_len ) { goto FAILED; }

    cudaSetDevice(this_gpu);

    // transfer the signal to the gpu
    cuda_status = cudaMemcpyAsync(this_p_data_gpu, p_signal, signal_len, cudaMemcpyHostToDevice, this_stream);
    if ( cuda_status != cudaSuccess ) { goto FAILED; }

    // apply window function [each row]
    p_data_gpu = this_p_data_gpu;
    for ( uint32_t row = 0; row < this_num_rows; row++ )
    {
        cudak_apply_window<<< num_thread_blocks, num_threads_per_block, 0, this_stream >>>
          ((cuFloatComplex *)p_data_gpu, (cuFloatComplex *)p_data_gpu, (float *)this_p_window_gpu, this_fft_size);

        p_data_gpu += this_complex_row_len;
    }

    // execute forward fft [all rows in parallel - see creation of fft plan]
    p_data_gpu = this_p_data_gpu;
    fft_status = cufftExecC2C(this_fft_handle, (cufftComplex *)p_data_gpu, (cufftComplex *)p_data_gpu, CUFFT_FORWARD);
    if ( fft_status != CUFFT_SUCCESS ) { goto FAILED; }

    // execute fft shift [each row]
    p_data_gpu = this_p_data_gpu;
    for ( uint32_t row = 0; row < this_num_rows; row++ )
    {
        cudak_fft_shift<<< num_thread_blocks, num_threads_per_block, 0, this_stream >>>
          ((cuFloatComplex *)p_data_gpu, this_fft_size);

        p_data_gpu += this_complex_row_len;
    }

    // compute absolute value [each row]
    p_data_gpu = this_p_data_gpu;
    for ( uint32_t row = 0; row < this_num_rows; row++ )
    {
        cudak_abs<<< num_thread_blocks, num_threads_per_block, 0, this_stream >>>
          ((float *)p_data_gpu, (cuFloatComplex *)p_data_gpu, this_fft_size);

        p_data_gpu += this_complex_row_len;
    }

    // normalize with maximum [each row]
    p_data_gpu = this_p_data_gpu;

    for ( uint32_t row = 0; row < this_num_rows; row++ )
    {
        // get maximum via reduction
        uint32_t num_maxred_items = this_fft_size;

        cudak_max_reduce<<< num_thread_blocks, num_threads_per_block, maxred_shmem_size, this_stream >>>
          ((float *)this_p_float_row_gpu, (float *)p_data_gpu, num_maxred_items);

        num_maxred_items >>= num_threads_per_block_bits;
        num_thread_blocks  = dim3(num_maxred_items >> num_threads_per_block_bits);

        while ( num_maxred_items > (1 << num_threads_per_block_bits) )
        {
            cudak_max_reduce<<< num_thread_blocks, num_threads_per_block, maxred_shmem_size, this_stream >>>
              ((float *)this_p_float_row_gpu, (float *)this_p_float_row_gpu, num_maxred_items);

            num_maxred_items >>= num_threads_per_block_bits;
            num_thread_blocks  = dim3(num_maxred_items >> num_threads_per_block_bits);
        }

        if ( num_thread_blocks.x == 0 ) { num_thread_blocks = dim3(1); }

        cudak_max_reduce<<< num_thread_blocks, num_threads_per_block, maxred_shmem_size, this_stream >>>
          ((float *)this_p_float_row_gpu, (float *)this_p_float_row_gpu, num_maxred_items);

        num_thread_blocks     = dim3(this_fft_size >> num_threads_per_block_bits);
        num_threads_per_block = dim3(            1 << num_threads_per_block_bits);

        // normalize with maximum
        cudak_norm_max<<< num_thread_blocks, num_threads_per_block, 0, this_stream >>>
          ((float *)p_data_gpu, (float *)p_data_gpu, this_fft_size, (float *)this_p_float_row_gpu);

        p_data_gpu += this_float_row_len;
    }

    // compute nlog10 [each row]
    p_data_gpu = this_p_data_gpu;
    for ( uint32_t row = 0; row < this_num_rows; row++ )
    {
        cudak_nlog10<<< num_thread_blocks, num_threads_per_block, 0, this_stream >>>
          ((float *)p_data_gpu, (float *)p_data_gpu, this_fft_size, /*n*/10.0f);

        p_data_gpu += this_float_row_len;
    }

    // sum all rows and average
    ok = this->zero_float_row((float *)this_p_float_row_gpu);
    if ( ! ok ) { goto FAILED; }

    p_data_gpu = this_p_data_gpu;
    for ( uint32_t row = 0; row < this_num_rows; row++ )
    {
        cudak_sum_rows<<< num_thread_blocks, num_threads_per_block, 0, this_stream >>>
          ((float *)this_p_float_row_gpu, (float *)this_p_float_row_gpu, (float *)p_data_gpu, this_fft_size);

        p_data_gpu += this_float_row_len;
    }

    cudak_average<<< num_thread_blocks, num_threads_per_block, 0, this_stream >>>
      ((float *)this_p_float_row_gpu, (float *)this_p_float_row_gpu, this_fft_size);

    // transfer the psd estimates to the host
    cuda_status = cudaMemcpyAsync(p_psd_est_buffer, this_p_float_row_gpu, psd_est_buffer_len, cudaMemcpyDeviceToHost,
                                  this_stream);
    if ( cuda_status != cudaSuccess ) { goto FAILED; }

    cudaStreamSynchronize(this_stream);

    return ( true );

FAILED:
    return ( false );
}

//-- private methods -------------------------------------------------------------------------------------------------------

// zero out a row of floats
bool cuda_psd_est_proc::zero_float_row( float *p_row_gpu )
{
    cudaError_t cuda_status = cudaMemcpyAsync(p_row_gpu, this_zeros_row.data(), this_float_row_len,
                                              cudaMemcpyHostToDevice, this_stream);
    if ( cuda_status != cudaSuccess ) { goto FAILED; }

    return ( true );

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// generate the window function
bool cuda_psd_est_proc::generate_window( void )
{
    float      *p_window_host = nullptr;
    float       m             = (float)(this_num_items - 1);
    cudaError_t status        = cudaSuccess;

    // allocate a host buffer to hold the window
    p_window_host = (float *)malloc(this_window_len);
    if ( p_window_host == nullptr ) { goto FAILED; }

    // generate a hamming window
    for ( uint32_t i = 0; i < this_num_items; i++ ) { p_window_host[i] = (0.54 - 0.46 * cos((2 * M_PI * i) / m)); }

    // transfer the window to the gpu
    status = cudaMemcpyAsync(this_p_window_gpu, p_window_host, this_window_len, cudaMemcpyHostToDevice, this_stream);
    if ( status != cudaSuccess ) { goto FAILED; }

    // clean up
    free(p_window_host);

    return ( true );

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
