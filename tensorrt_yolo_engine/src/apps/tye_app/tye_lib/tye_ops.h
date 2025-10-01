#ifndef INCLUDE_TYE_OPS_H
#define INCLUDE_TYE_OPS_H

//--------------------------------------------------------------------------------------------------------------------------

#include "cuda_spectrogram.hpp"
#include "rssi.hpp"
#include "tye_includes.h"
#include "tye_types.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class tye_ops
{
public: //==================================================================================================================

    // constructor(s) / destructor
    tye_ops( uint32_t gpu, std::string engine_path, uint32_t engine_id, uint32_t fft_size );
   ~tye_ops( void );

    // public methods
    void    print_io_tensors( void );
    void    warmup_inference( uint32_t num_iter );
    double  preprocess_samples( uint8_t *p_samples, uint32_t samples_len, float pixel_min_val, float pixel_max_val,
                                cv::Mat &image_out_host, float *p_spectrogram_out_host, double *p_spectrogram_usec );
    double  preprocess_image( cv::cuda::GpuMat &image );
    double  run_inference( void );
    double  get_detections( float score_threshold, float nms_threshold, double *p_nms_usec,
                            std::vector<tye_types::detection> &detections );
    double  calc_detections_rssi( float *p_spectrogram_data, int32_t width, int32_t height,
                                  std::vector<tye_types::detection> &detections, float power_offset );
    cv::Mat get_image_with_detections( cv::Mat &image, uint32_t false_detect_w, uint32_t false_detect_h,
                                       std::vector<tye_types::detection> &detections );

private: //=================================================================================================================

    // private constants
    const std::string NAME = std::string("TYE_OPS");

    // private types
    typedef struct
    {
        std::string            name;
        nvinfer1::TensorIOMode io_type;
        nvinfer1::DataType     data_type;
        nvinfer1::Dims         shape;
        uint32_t               num_batches;
        uint32_t               batch_len;
        void                  *p_buffer;

    } io_tensor;

    // private variables
    uint32_t                     this_gpu;
    std::string                  this_engine_path;
    uint32_t                     this_engine_id;
    uint32_t                     this_fft_size;
    nvinfer1::IRuntime          *this_p_runtime;
    nvinfer1::ICudaEngine       *this_p_engine;
    nvinfer1::IExecutionContext *this_p_context;
    cudaStream_t                 this_stream;
    cv::cuda::Stream             this_cv_stream;
    float                       *this_p_host_detections;
    uint8_t                     *this_p_input_tmp;
    tye_ops::io_tensor          *this_p_input_tensor;
    tye_ops::io_tensor          *this_p_output_tensor;
    spectrogram                 *this_p_spectrogram;
    rssi                        *this_p_rssi;

    std::vector<tye_ops::io_tensor> this_io_tensors;
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_TYE_OPS_THREAD_H
