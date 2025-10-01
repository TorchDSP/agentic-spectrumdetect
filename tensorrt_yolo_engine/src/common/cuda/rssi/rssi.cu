#include "rssi.hpp"

#define BLOCK_SIZE 512

#define CUDA_ERR_CHK(ans) { cuda_sucess_chk((ans), __FILE__, __LINE__); }
inline void cuda_sucess_chk(cudaError_t code, const char* file, int line, bool abort=true){
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void calculateMultipleRSSI_kernel(float* spectrogram, int width, int height, float* boxes, int numBoxes, float powerOffset, float* rssiResults) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numBoxes) {
        float box_x = boxes[idx * 4];
        float box_y = boxes[idx * 4 + 1];
        float box_w = boxes[idx * 4 + 2];
        float box_h = boxes[idx * 4 + 3];
        
        int x_start = static_cast<int>(box_x * width- (box_w * width / 2));
        int y_start = static_cast<int>(box_y * height - (box_h * height / 2));
        int box_width = static_cast<int>(box_w * width);
        int box_height = static_cast<int>(box_h * height);
        
        x_start = max(0, min(x_start, width - 1));
        y_start = max(0, min(y_start, height -1));
        box_width = min(box_width, width - x_start);
        box_height = min(box_height, height - y_start);
        
        float sum = 0.0f;
        int count = 0;
        
        for (int y = y_start; y < y_start + box_height; y++) {
            for (int x = x_start; x < x_start + box_width; x++) {
                sum += spectrogram[y * width + x];
                count++;
            }
        }
        
        if (count > 0) {
            float avgPower = sum / count;
            rssiResults[idx] = avgPower + powerOffset;
        }
        else {
            rssiResults[idx] = -INFINITY;
        }
    }
}      

rssi::rssi( unsigned int gpu, int fft_size, int num_boxes, cudaStream_t stream )
{
    cudaSetDevice(gpu);

    d_gpu = gpu;
    d_stream = stream;
    numBoxes = num_boxes;
    fftSize = fft_size;
    fftSizeSquared = fftSize * fftSize; 
    //if (numBoxes > 1)
      //{
    blocks = (numBoxes + BLOCK_SIZE - 1) / BLOCK_SIZE;
      //}
    //else
      //{
        //blocks = 2;
      //}

    CUDA_ERR_CHK(cudaMalloc((void **)&d_spectrogram, fftSizeSquared * sizeof(float)));
    CUDA_ERR_CHK(cudaMalloc((void **)&d_boxes, numBoxes * 4 * sizeof(float)));
    CUDA_ERR_CHK(cudaMalloc((void **)&d_rssiResults, numBoxes * sizeof(float)));   
    
    return; 
}

    
rssi::~rssi( void )
{
    cudaSetDevice(d_gpu);

    CUDA_ERR_CHK( cudaFree(d_spectrogram) ); 
    CUDA_ERR_CHK( cudaFree(d_boxes));
    CUDA_ERR_CHK( cudaFree(d_rssiResults));

    return;
}


bool rssi::resize_buffers(int new_num_boxes) {
    if (new_num_boxes == numBoxes) {
        return true;
    }

    cudaSetDevice(d_gpu);

    CUDA_ERR_CHK(cudaFree(d_boxes));
    CUDA_ERR_CHK(cudaFree(d_rssiResults));

    CUDA_ERR_CHK(cudaMalloc((void **)&d_boxes, new_num_boxes * 4 * sizeof(float)));
    CUDA_ERR_CHK(cudaMalloc((void **)&d_rssiResults, new_num_boxes * sizeof(float)));

    numBoxes = new_num_boxes;
    blocks = (numBoxes + BLOCK_SIZE - 1) / BLOCK_SIZE;

    return true;
}


float*  rssi::calculateMultipleRSSI(  float* h_spectrogram, int width, int height, float* h_boxes, int numBoxes, float powerOffset, float* h_rssiResults)
{
    cudaSetDevice(d_gpu);

    CUDA_ERR_CHK(cudaMemcpyAsync(d_spectrogram, h_spectrogram, width * height * sizeof(float), cudaMemcpyHostToDevice, d_stream));
    CUDA_ERR_CHK(cudaMemcpyAsync(d_boxes, h_boxes, numBoxes * 4 * sizeof(float), cudaMemcpyHostToDevice, d_stream));

    calculateMultipleRSSI_kernel<<<blocks, BLOCK_SIZE, 0, d_stream>>>(d_spectrogram, width, height, d_boxes, numBoxes, powerOffset, d_rssiResults);
    
    CUDA_ERR_CHK(cudaMemcpyAsync(h_rssiResults, d_rssiResults, numBoxes * sizeof(float),cudaMemcpyDeviceToHost, d_stream));

    cudaStreamSynchronize(d_stream);
 
    return  h_rssiResults ;
}
              

