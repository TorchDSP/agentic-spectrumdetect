/*
Cuda code to accelerate spectrogram image creation
*/

#include "cuda_spectrogram.hpp"

#define BLOCK_SIZE 512
#define TRANSPOSE_BLOCK_DIM 16

// cuFFT API errors
#ifdef _CUFFT_H_
static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}
#endif

/* ---------- CPU helper functions below here ---------- */

#define CUFFT_ERR_CHK(ans) { cufft_success_chk((ans), __FILE__, __LINE__); }
inline void cufft_success_chk(cufftResult result, const char* file, int line, bool abort=true){
    if (result != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error in file '%s', line %d\n error %d: %s\nterminating!\n", __FILE__, __LINE__, result, _cudaGetErrorEnum(result));
        if (abort) exit(result);
    }
}

#define CUDA_ERR_CHK(ans) { cuda_sucess_chk((ans), __FILE__, __LINE__); }
inline void cuda_sucess_chk(cudaError_t code, const char* file, int line, bool abort=true){
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* ---------- CUDA kernels for spectrogram here ---------- */

__global__
void convertFloatToBF16(cufftComplex *input, __nv_bfloat162 * output, unsigned int size)
{
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (tid < size)
    {
        output[tid].x = __float2bfloat16(input[tid].x);
        output[tid].y = __float2bfloat16(input[tid].y);
    }
    
    return;
}        

__global__ 
void blackmanHarrisWindow( __nv_bfloat162 *sample, unsigned int fftSizeSquared, unsigned int fftSize )
{
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if ( tid < fftSizeSquared )
    {
        float shift   = ((tid % 2) == 0) ? -1.0f : 1.0f;
        float val     = (float)M_PI*(tid%fftSize)/fftSize;
        float window = 0.42f - 0.5f * cosf(2.0f * val) + 0.08f * cosf(4.0f * val);
        
        float real = __bfloat162float(sample[tid].x);
        float imag = __bfloat162float(sample[tid].y);
        
        real *= shift * window;
        imag *= shift * window;
        
        sample[tid].x = __float2bfloat16(real);
        sample[tid].y = __float2bfloat16(imag);
    }

    return;
}



__global__ 
void squareMagnitudesLog10AndTranspose( __nv_bfloat162 *signal_in, __nv_bfloat16 *output, int width )
{

    __shared__ __nv_bfloat16 tile[TRANSPOSE_BLOCK_DIM][TRANSPOSE_BLOCK_DIM + 1];
    
    int x = blockIdx.x * TRANSPOSE_BLOCK_DIM + threadIdx.x;
    int y = blockIdx.y * TRANSPOSE_BLOCK_DIM + threadIdx.y;
    int index = y * width + x;
    
    if (x < width && y < width) {
        float real = __bfloat162float(signal_in[index].x);
        float imag = __bfloat162float(signal_in[index].y);
        
        float mag_squared = real * real + imag * imag;
        float log_value = 10.0f * log10f(max(mag_squared, 1e-10f));
        
        tile[threadIdx.y][threadIdx.x] = __float2bfloat16(log_value);
    }
    
    __syncthreads();
    
    x = blockIdx.y * TRANSPOSE_BLOCK_DIM + threadIdx.x;
    //y = blockIdx.x * TRANSPOSE_BLOCK_DIM + threadIdx.y; //Clockwise
    y = width - 1 - (blockIdx.x * TRANSPOSE_BLOCK_DIM + threadIdx.y); //Counter Clockwise
    index = y * width + x;
    
    //if (x < width && y < width) {
    if (x < width && y >= 0 && y < width) {
        output[index] = tile[threadIdx.x][threadIdx.y];
    }

    return;
}        

__global__
void convertBF16ToFloat(__nv_bfloat16 *input, float *output, unsigned int size)
{
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (tid < size)
    {
        output[tid] = __bfloat162float(input[tid]);
    }
    
    return;
}
   
__global__
void convertSpectrogramToBlackHotImage(__nv_bfloat16 *spectrogram_in, unsigned char *image_out, unsigned int size, float min_val, float max_val)
{
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < size) {
	float pixel_value = __bfloat162float(spectrogram_in[tid]);

        pixel_value = 1.0f - ((pixel_value - min_val) / (max_val - min_val));


        unsigned char value = (unsigned char)(255 * min(max(pixel_value, 0.0f), 1.0f));

        unsigned int dst_idx = tid * 3;
        image_out[dst_idx] = value;
        image_out[dst_idx + 1] = value;
        image_out[dst_idx + 2] = value;
    }
    
    return;
}    
    

spectrogram::spectrogram( unsigned int gpu, long long int fft_size, cudaStream_t stream )
{
    cudaSetDevice(gpu);

    d_gpu = gpu;
    d_stream = stream;
    fftSize = fft_size;
    fftSizeSquared = fftSize * fftSize; 
    blocks = (fftSizeSquared + BLOCK_SIZE -1) / BLOCK_SIZE;

    CUDA_ERR_CHK(cudaMalloc((void **)&d_float_signal, fftSizeSquared * sizeof(cufftComplex)));
    CUDA_ERR_CHK(cudaMalloc((void **)&d_signal, fftSizeSquared * sizeof(__nv_bfloat162)));
    CUDA_ERR_CHK(cudaMalloc((void **)&d_output, fftSizeSquared * sizeof(__nv_bfloat16)));
    CUDA_ERR_CHK(cudaMalloc((void **)&d_float_output, fftSizeSquared * sizeof(float)));
    CUDA_ERR_CHK(cudaMalloc((void **)&d_image_output, fftSizeSquared * 3 * sizeof(unsigned char)));


    //int n[1] = {fft_size};
    long long int n[1] = {fft_size};
    size_t workSize;
    
    CUFFT_ERR_CHK(cufftCreate(&fft_plan));
    
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    if (deviceProp.major < 8) {
        fprintf(stderr, "Warning: This  GPU may not support bfloat16 natively\n");
    }    
    
    CUFFT_ERR_CHK(cufftXtMakePlanMany(
        fft_plan, 
        1, 
        n, 
        NULL, 
        0, 
        0, 
        CUDA_C_16BF, 
        NULL, 
        0, 
        0, 
        CUDA_C_16BF, 
        fft_size, 
        &workSize, 
        CUDA_C_16BF
    ));
    CUFFT_ERR_CHK(cufftSetStream(fft_plan, d_stream));

    return;
}  
                                                                                                         
spectrogram::~spectrogram( void )
{
    cudaSetDevice(d_gpu);

    CUFFT_ERR_CHK( cufftDestroy(fft_plan) );
    CUDA_ERR_CHK( cudaFree(d_float_signal) );
    CUDA_ERR_CHK( cudaFree(d_signal) ); 
    CUDA_ERR_CHK( cudaFree(d_output) ); 
    CUDA_ERR_CHK( cudaFree(d_float_output));
    CUDA_ERR_CHK( cudaFree(d_image_output));

    return;
}
      
float*  spectrogram::generate( cufftComplex *samples_in_host, float *spectrogram_out_host, float min_val, float max_val )
{
    cudaSetDevice(d_gpu);

    CUDA_ERR_CHK(cudaMemcpyAsync(d_float_signal, samples_in_host, fftSizeSquared * sizeof(cufftComplex), cudaMemcpyHostToDevice, d_stream));
    
    convertFloatToBF16<<<blocks, BLOCK_SIZE, 0, d_stream>>>(d_float_signal, d_signal, fftSizeSquared);

    blackmanHarrisWindow<<<blocks, BLOCK_SIZE, 0, d_stream>>>(d_signal, fftSizeSquared, fftSize);
    
    CUFFT_ERR_CHK(cufftXtExec(fft_plan, d_signal, d_signal, CUFFT_FORWARD));
    
    dim3 dimBlock(TRANSPOSE_BLOCK_DIM, TRANSPOSE_BLOCK_DIM);
    dim3 dimGrid((fftSize + TRANSPOSE_BLOCK_DIM -1) / TRANSPOSE_BLOCK_DIM, (fftSize + TRANSPOSE_BLOCK_DIM -1) / TRANSPOSE_BLOCK_DIM);
    
    squareMagnitudesLog10AndTranspose<<<dimGrid, dimBlock, 0, d_stream>>>(d_signal, d_output, fftSize);

    convertSpectrogramToBlackHotImage<<<blocks, BLOCK_SIZE, 0, d_stream>>>(d_output, d_image_output, fftSizeSquared, min_val, max_val);

    if ( spectrogram_out_host != nullptr )
    {
        convertBF16ToFloat<<<blocks, BLOCK_SIZE, 0, d_stream>>>(d_output, d_float_output, fftSizeSquared); 
        CUDA_ERR_CHK(cudaMemcpyAsync(spectrogram_out_host, d_float_output, fftSizeSquared * sizeof(float),
                                     cudaMemcpyDeviceToHost, d_stream));
                                                                         
    }

    cudaStreamSynchronize(d_stream);
 
    return  d_float_output ;
}
