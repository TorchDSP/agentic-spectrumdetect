#ifndef CUDA_SPECTROGRAM_HPP
#define CUDA_SPECTROGRAM_HPP


#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cufftXt.h>
#include <cufft.h>

class spectrogram
{
    public:

        spectrogram( unsigned int gpu, long long int fft_size, cudaStream_t stream );
       ~spectrogram( void );

        //unsigned char* generate( cufftComplex *samples_in_host, float *spectrogram_out_host, float min_val = -60.0f, float max_val = 0.0f );
        float* generate( cufftComplex *samples_in_host, float *spectrogram_out_host, float min_val = -60.0f, float max_val = 0.0f  );

        unsigned char* d_image_output;

    private:

        unsigned int d_gpu;
        unsigned int blocks;
        cudaStream_t d_stream;
        cufftComplex *d_float_signal;
        __nv_bfloat162 *d_signal;
        __nv_bfloat16 *d_output;
        int fftSize;
        int fftSizeSquared;
        float *d_float_output;
	//unsigned char* d_image_output;
	//int size;
        cufftHandle fft_plan;
};

#endif // CUDA_SPECTROGRAM_HPP
