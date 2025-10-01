#ifndef RSSI_HPP
#define RSSI_HPP


#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


class rssi
{
    public:

        rssi( unsigned int gpu, int fft_size, int num_boxes, cudaStream_t stream );
       ~rssi( void );

        float* calculateMultipleRSSI( float* h_spectrogram, int width, int height, float* h_boxes, int numBoxes, float powerOffset, float* h_rssiResults );

	bool resize_buffers(int new_num_boxes);


    private:

        unsigned int d_gpu;
        unsigned int blocks;
        cudaStream_t d_stream;
        int fftSize;
	int fftSizeSquared;
        int numBoxes;
        float* d_spectrogram;
        float* d_boxes;
        float* d_rssiResults;
};

#endif // RSSI_HPP
