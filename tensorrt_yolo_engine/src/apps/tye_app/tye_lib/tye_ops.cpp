//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "cuda_preprocess_image.h"
#include "tye_ops.h"

//-- note ------------------------------------------------------------------------------------------------------------------

// The following projects were leveraged to help produce this source code
//
//   https://github.com/spacewalk01/yolov11-tensorrt
//   https://github.com/cyrusbehr/tensorrt-cpp-api

//-- tensor-rt inference logger --------------------------------------------------------------------------------------------

class Logger : public nvinfer1::ILogger
{
    void log( Severity severity, const char* msg ) noexcept override
    {
        if ( severity <= Severity::kWARNING ) { std::cout << msg << std::endl; }
    }

} tye_ops_logger;

//-- constructor(s) --------------------------------------------------------------------------------------------------------

tye_ops::tye_ops( uint32_t gpu, std::string engine_path, uint32_t engine_id, uint32_t fft_size )
{
    cudaError_t cuda_status = cudaSuccess;

    // initialize variables
    this_gpu               = gpu;
    this_engine_path       = engine_path;
    this_engine_id         = engine_id;
    this_fft_size          = fft_size;
    this_p_runtime         = nullptr;
    this_p_engine          = nullptr;
    this_p_context         = nullptr;
    this_stream            = {};
    this_cv_stream         = {};
    this_p_host_detections = nullptr;
    this_p_input_tmp       = nullptr;
    this_p_input_tensor    = nullptr;
    this_p_output_tensor   = nullptr;
    this_p_spectrogram     = nullptr;
    this_p_rssi            = nullptr;
    
    this_io_tensors.clear();

    try
    {
        // load the tensor-rt yolo engine
        size_t        engine_size = 0;
        std::ifstream engine_stream(engine_path, std::ios::binary);

        engine_stream.seekg(0, std::ios::end);
        engine_size = engine_stream.tellg();
        engine_stream.seekg(0, std::ios::beg);

        std::unique_ptr<char[]> engine_payload(new char[engine_size]);

        engine_stream.read(engine_payload.get(), engine_size);
        engine_stream.close();

        // create the tensor-rt yolo engine runtime and execution context
        cudaSetDevice(this_gpu);

        this_p_runtime = nvinfer1::createInferRuntime(tye_ops_logger);
        if ( this_p_runtime == nullptr )
        {
            throw std::runtime_error(tye_ops::NAME + " [EXCEPTION] CALL nvinfer1::createInferRuntime()");
        }
        this_p_runtime->setMaxThreads(1);

        this_p_engine = this_p_runtime->deserializeCudaEngine(engine_payload.get(), engine_size);
        if ( this_p_engine == nullptr )
        {
            throw std::runtime_error(tye_ops::NAME + " [EXCEPTION] CALL runtime->deserializeCudaEngine()");
        }

        this_p_context = this_p_engine->createExecutionContext();
        if ( this_p_context == nullptr )
        {
            throw std::runtime_error(tye_ops::NAME + " [EXCEPTION] CALL engine->createExecutionContext()");
        }

        // create a cuda stream for host/GPU data transfers and inference
        cuda_status = cudaStreamCreate(&this_stream);
        if ( cuda_status != cudaSuccess )
        {
            throw std::runtime_error(tye_ops::NAME + " [EXCEPTION] CREATING GPU/CUDA STREAM");
        }
        this_cv_stream = cv::cuda::StreamAccessor::wrapStream(this_stream);

        // collect I/O tensor information and allocate tensor-specific resources
        uint32_t t_num_inputs  = 0;
        uint32_t t_num_outputs = 0;

        for ( uint32_t i = 0; i < this_p_engine->getNbIOTensors(); ++i )
        {
            std::string            t_name    = this_p_engine->getIOTensorName(i);
            nvinfer1::TensorIOMode t_io_type = this_p_engine->getTensorIOMode(t_name.c_str());

            if ( (t_io_type != nvinfer1::TensorIOMode::kINPUT) && (t_io_type != nvinfer1::TensorIOMode::kOUTPUT) )
            {
                throw std::runtime_error(tye_ops::NAME + " [EXCEPTION] TENSOR I/O TYPE MUST BE INPUT OR OUTPUT");
            }

            if      ( t_io_type == nvinfer1::TensorIOMode::kINPUT  ) { t_num_inputs++;  }
            else if ( t_io_type == nvinfer1::TensorIOMode::kOUTPUT ) { t_num_outputs++; }

            if      ( t_num_inputs  > 1 ) { throw std::runtime_error(tye_ops::NAME + " [EXCEPTION] TOO MANY INPUTS" ); }
            else if ( t_num_outputs > 1 ) { throw std::runtime_error(tye_ops::NAME + " [EXCEPTION] TOO MANY OUTPUTS"); }

            nvinfer1::DataType t_data_type   = this_p_engine->getTensorDataType(t_name.c_str());
            nvinfer1::Dims     t_shape       = this_p_engine->getTensorShape(t_name.c_str());
            uint32_t           t_num_batches = t_shape.d[0];
            uint32_t           t_batch_len   = 1;
            uint32_t           t_batch_bytes = 0;

            for ( uint32_t j = 1; j < t_shape.nbDims; j++ ) { t_batch_len *= t_shape.d[j]; }
            t_batch_bytes = (uint32_t)(t_num_batches * t_batch_len * sizeof(float));

            if ( (t_io_type == nvinfer1::TensorIOMode::kINPUT) && (this_p_host_detections == nullptr) )
            {
                cuda_status = cudaMallocAsync(&this_p_input_tmp, t_batch_bytes, this_stream);
                if ( cuda_status != cudaSuccess )
                {
                    throw std::runtime_error(tye_ops::NAME + " [EXCEPTION] ALLOCATE GPU/CUDA INPUT TMP BUFFER");
                }

                cuda_status = cudaMallocHost(&this_p_host_detections, t_batch_bytes);
                if ( this_p_host_detections == nullptr )
                {
                    throw std::runtime_error(tye_ops::NAME + " [EXCEPTION] ALLOCATE HOST DETECTIONS BUFFER");
                }
            }

            void *t_p_buffer = nullptr;

            cuda_status = cudaMallocAsync(&t_p_buffer, t_batch_bytes, this_stream);
            if ( cuda_status != cudaSuccess )
            {
                throw std::runtime_error(tye_ops::NAME + " [EXCEPTION] ALLOCATE GPU/CUDA BUFFER");
            }
            cudaStreamSynchronize(this_stream);

            tye_ops::io_tensor io_tensor = { .name        = t_name,        .io_type   = t_io_type,
                                             .data_type   = t_data_type,   .shape     = t_shape,
                                             .num_batches = t_num_batches, .batch_len = t_batch_len,
                                             .p_buffer    = t_p_buffer
                                           };
            this_io_tensors.push_back(io_tensor);
        }

        if      ( t_num_inputs  == 0 ) { throw std::runtime_error(tye_ops::NAME + " [EXCEPTION] NO INPUTS" ); }
        else if ( t_num_outputs == 0 ) { throw std::runtime_error(tye_ops::NAME + " [EXCEPTION] NO OUTPUTS"); }

        // at this point, there is exactly one input tensor and one output tensor...set a pointer to
        // each so we don't have to do a lookups later
        uint32_t num_io_tensors = (uint32_t)this_io_tensors.size();

        for ( uint32_t i = 0; i < num_io_tensors; i++ )
        {
            if ( this_io_tensors.at(i).io_type == nvinfer1::TensorIOMode::kINPUT )
            {
                this_p_input_tensor = &this_io_tensors.at(i);
            }
            else { this_p_output_tensor = &this_io_tensors.at(i); }
        }

        // create the spectrogram generator and RSSI calculator
        this_p_spectrogram = new spectrogram(this_gpu, this_fft_size, this_stream);
        if ( this_p_spectrogram == nullptr ) { throw std::runtime_error(tye_ops::NAME + " [EXCEPTION] CREATE SPECTROGRAM"); }

        this_p_rssi = new rssi(this_gpu, this_fft_size, /*num_boxes*/1, this_stream);
        if ( this_p_rssi == nullptr ) { throw std::runtime_error(tye_ops::NAME + " [EXCEPTION] CREATE RSSI CALCULATOR"); }
    }
    catch ( ... ) { throw std::runtime_error(tye_ops::NAME + " [EXCEPTION] LOAD TRT YOLO ENGINE"); }

    return;    
}

//-- destructor ------------------------------------------------------------------------------------------------------------

tye_ops::~tye_ops( void )
{
    // clean up
    delete this_p_context;
    delete this_p_engine;
    delete this_p_runtime;
    delete this_p_spectrogram;
    delete this_p_rssi;

    cudaStreamSynchronize(this_stream);
    cudaStreamDestroy(this_stream);
    
    uint32_t num_io_tensors = (uint32_t)this_io_tensors.size();

    for ( uint32_t i = 0; i < num_io_tensors; i++ )
    {
        if ( this_io_tensors.at(i).p_buffer != nullptr ) { cudaFree(this_io_tensors.at(i).p_buffer); }
    }

    cudaFree(this_p_input_tmp);
    cudaFreeHost(this_p_host_detections);

    return;
}

//-- public methods --------------------------------------------------------------------------------------------------------

// print input and output tensor information
void tye_ops::print_io_tensors( void )
{
    uint32_t num_io_tensors = (uint32_t)this_io_tensors.size();

    for ( uint32_t i = 0; i < num_io_tensors; i++ )
    {
        std::cout << ">> TENSOR " << i << " NAME [" << this_io_tensors.at(i).name << "] " << std::flush;

        nvinfer1::TensorIOMode tensor_iotype = this_io_tensors.at(i).io_type;
        nvinfer1::DataType     tensor_dtype  = this_io_tensors.at(i).data_type;

        std::cout << "IOTYPE " << std::flush;
        if      ( tensor_iotype == nvinfer1::TensorIOMode::kINPUT  ) { std::cout << "[INPUT] "  << std::flush; }
        else if ( tensor_iotype == nvinfer1::TensorIOMode::kOUTPUT ) { std::cout << "[OUTPUT] " << std::flush; }

        std::cout << "DTYPE " << std::flush;
        if      ( tensor_dtype == nvinfer1::DataType::kFLOAT ) { std::cout << "[FP32] " << std::flush; }
        else if ( tensor_dtype == nvinfer1::DataType::kHALF  ) { std::cout << "[FP16] " << std::flush; }
        else if ( tensor_dtype == nvinfer1::DataType::kINT8  ) { std::cout << "[INT8] " << std::flush; }

        std::cout << "SHAPE [" << std::flush;
        for ( uint32_t j = 0; j < this_io_tensors.at(i).shape.nbDims; j++ )
        {
            std::cout << this_io_tensors.at(i).shape.d[j] << std::flush;
            if ( j < (this_io_tensors.at(i).shape.nbDims - 1) ) { std::cout << ", " << std::flush; }
        }
        std::cout << "]" << std::endl << std::flush;
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// warm-up inference by running it for a requested number of iterations
void tye_ops::warmup_inference( uint32_t num_iter )
{
    for ( uint32_t i = 0; i < num_iter; i++ ) { this->run_inference(); }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// pre-inference processing when samples are the input
double tye_ops::preprocess_samples( uint8_t *p_samples, uint32_t samples_len, float pixel_min_val, float pixel_max_val,
                                    cv::Mat &image_out_host, float *p_spectrogram_out_host, double *p_spectrogram_usec )
{
    auto start_time = std::chrono::system_clock::now();

    // create a spectrogram image from the samples (stored in GPU memory)
    float *p_sg_data = this_p_spectrogram->generate((cufftComplex *)p_samples, p_spectrogram_out_host, pixel_min_val,
                                                    pixel_max_val);

    auto sg_end_time   = std::chrono::system_clock::now();
   *p_spectrogram_usec = (double)std::chrono::duration_cast<std::chrono::microseconds>(sg_end_time - start_time).count();

#ifdef TYE_FILE_PROCESSOR

    cv::cuda::GpuMat image_gpu = cv::cuda::GpuMat(this_fft_size, this_fft_size, CV_32F, p_sg_data);

    cv::cuda::normalize(image_gpu, image_gpu, 0, 255, cv::NORM_MINMAX, -1, /*mask*/cv::noArray(), this_cv_stream);
    image_gpu.convertTo(image_gpu, CV_8U, this_cv_stream);
    cv::cuda::cvtColor(image_gpu, image_gpu, cv::COLOR_GRAY2BGR, /*dcn*/0, this_cv_stream);
    cv::cuda::bitwise_not(image_gpu, image_gpu, /*mask*/cv::noArray(), this_cv_stream);

#else // TYE_STREAM_PROCESSOR, etc

    cv::cuda::GpuMat image_gpu(this_fft_size, this_fft_size, CV_8UC3, this_p_spectrogram->d_image_output);

#endif

    image_gpu.download(image_out_host, this_cv_stream);
    this_cv_stream.waitForCompletion();
    
    this->preprocess_image(image_gpu);

    auto   end_time = std::chrono::system_clock::now();
    double rt_usec  = (double)std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    return ( rt_usec );
}

//--------------------------------------------------------------------------------------------------------------------------

// image pre-processing when the image is in GPU memory
double tye_ops::preprocess_image( cv::cuda::GpuMat &image )
{
    auto start_time = std::chrono::system_clock::now();

    int32_t tensor_input_height = this_p_input_tensor->shape.d[2];
    int32_t tensor_input_width  = this_p_input_tensor->shape.d[3];

    cuda_preprocess_image_gpu(image.ptr(), image.cols, image.rows, (float *)this_p_input_tensor->p_buffer,
                              tensor_input_width, tensor_input_height, this_stream);

    auto   end_time = std::chrono::system_clock::now();
    double rt_usec  = (double)std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    return ( rt_usec );
}

//--------------------------------------------------------------------------------------------------------------------------

// kick off inference
double tye_ops::run_inference( void )
{
    auto     start_time     = std::chrono::system_clock::now();
    uint32_t num_io_tensors = (uint32_t)this_io_tensors.size();
    bool     ok             = false;

    for ( uint32_t i = 0; i < num_io_tensors; i++ )
    {
        tye_ops::io_tensor *p_io_tensor = &this_io_tensors.at(i);

        ok = this_p_context->setTensorAddress(p_io_tensor->name.c_str(), p_io_tensor->p_buffer);
        if ( ! ok ) { throw std::runtime_error(tye_ops::NAME + " [EXCEPTION] CALL context->setTensorAddress()"); }
    }

    ok = this_p_context->enqueueV3(this_stream);
    if ( ! ok ) { throw std::runtime_error(tye_ops::NAME + " [EXCEPTION] CALL context->enqueueV3()"); }

    auto   end_time = std::chrono::system_clock::now();
    double rt_usec  = (double)std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    return ( rt_usec );
}

//--------------------------------------------------------------------------------------------------------------------------

// get detection information (post-inference)
double tye_ops::get_detections( float score_threshold, float nms_threshold, double *p_nms_usec,
                                std::vector<tye_types::detection> &detections )
{
    auto start_time = std::chrono::system_clock::now();

    // transfer detections from GPU to host
    cudaError_t cuda_status = cudaMemcpyAsync(this_p_host_detections, this_p_output_tensor->p_buffer,
                                              this_p_output_tensor->batch_len, cudaMemcpyDeviceToHost,
                                              this_stream);
    if ( cuda_status != cudaSuccess )
    {
        throw std::runtime_error(tye_ops::NAME + " [EXCEPTION] GPU TO HOST TRANSFER OF DETECTIONS");
    }
    cudaStreamSynchronize(this_stream);

    // create a bounding box for all detections over the confidence score threshold
    int32_t rows = this_p_output_tensor->shape.d[1];
    int32_t cols = this_p_output_tensor->shape.d[2];

    cv::Mat detections_out(rows, cols, CV_32F, this_p_host_detections);

    std::vector<float>    scores           = std::vector<float>();
    std::vector<cv::Rect> detection_bboxes = std::vector<cv::Rect>();

    for ( uint32_t i = 0; i < detections_out.cols; i++ )
    {
        cv::Mat   classes_scores = detections_out.col(i).rowRange(4, rows);
        cv::Point class_id_point = cv::Point();
        double    score          = 0.0;

        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if ( score > score_threshold )
        {
            float x = detections_out.at<float>(0, i);
            float y = detections_out.at<float>(1, i);
            float w = detections_out.at<float>(2, i);
            float h = detections_out.at<float>(3, i);

            cv::Rect detection_bbox = cv::Rect();
      
            detection_bbox.x      = (int32_t)((x - 0.5 * w));
            detection_bbox.y      = (int32_t)((y - 0.5 * h));
            detection_bbox.width  = (int32_t)w;
            detection_bbox.height = (int32_t)h;

            scores.push_back(score);
            detection_bboxes.push_back(detection_bbox);
        }
    }

    auto nms_start = std::chrono::system_clock::now();

    // run the NMS algorithm to eliminate duplicate/overlapping bounding boxes
    std::vector<int32_t> nms_results = std::vector<int32_t>();
    cv::dnn::NMSBoxes(detection_bboxes, scores, score_threshold, nms_threshold, nms_results);

    uint32_t num_nms_results = (uint32_t)nms_results.size();

    for ( uint32_t i = 0; i < num_nms_results; i++ )
    {
        tye_types::detection detection = tye_types::detection();
        uint32_t             idx       = nms_results[i];

        detection.score = scores[idx];
        detection.bbox  = detection_bboxes[idx];

        detections.push_back(detection);
    }

    auto nms_end = std::chrono::system_clock::now();
   *p_nms_usec   = (double)std::chrono::duration_cast<std::chrono::microseconds>(nms_end - nms_start).count();

    auto   end_time = std::chrono::system_clock::now();
    double rt_usec  = (double)std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    return ( rt_usec );
}

//--------------------------------------------------------------------------------------------------------------------------

// calculate RSSI for each detection
double tye_ops::calc_detections_rssi( float *p_spectrogram_data, int32_t width, int32_t height,
                                      std::vector<tye_types::detection> &detections, float power_offset )
{
    auto    start_time     = std::chrono::system_clock::now();
    int32_t num_detections = detections.size();

    if ( num_detections > 0 )
    {
        if ( ! this_p_rssi->resize_buffers(num_detections) )
        {
            std::cerr << ">> " << tye_ops::NAME + "RESIZE RSSI BUFFERS [FAIL]" << std::endl;
            return ( 0.0 );
        }

        float *p_boxes        = new float[num_detections * 4];
        float *p_rssi_results = new float[num_detections];

        for ( int32_t i = 0; i < num_detections; i++ )
        {
            p_boxes[i*4]     = (detections[i].bbox.x + detections[i].bbox.width  / 2.0f) / width;
            p_boxes[i*4 + 1] = (detections[i].bbox.y + detections[i].bbox.height / 2.0f) / height;
            p_boxes[i*4 + 2] = (detections[i].bbox.width  / (float)width);
            p_boxes[i*4 + 3] = (detections[i].bbox.height / (float)height);
        }

        this_p_rssi->calculateMultipleRSSI(p_spectrogram_data, width, height, p_boxes, num_detections, power_offset,
                                           p_rssi_results);

        for ( int32_t i = 0; i < num_detections; i++ ) { detections[i].rssi = p_rssi_results[i]; }

        delete[] p_boxes;
        delete[] p_rssi_results;
    }

    auto   end_time = std::chrono::system_clock::now();
    double rt_usec  = (double)std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    return ( rt_usec );
}

//--------------------------------------------------------------------------------------------------------------------------

// produce an image that contains the detection bounding boxes
cv::Mat tye_ops::get_image_with_detections( cv::Mat &image, uint32_t false_detect_w, uint32_t false_detect_h,
                                            std::vector<tye_types::detection> &detections )
{
    float    tensor_input_height = (float)this_p_input_tensor->shape.d[2];
    float    tensor_input_width  = (float)this_p_input_tensor->shape.d[3];
    float    ratio_height        = (float)(tensor_input_height / (float)image.rows);
    float    ratio_width         = (float)(tensor_input_width  / (float)image.cols);
    uint32_t num_detections      = (uint32_t)detections.size();

    for ( uint32_t i = 0; i < num_detections; i++ )
    {
        cv::Rect d_bbox = detections[i].bbox;

        if ( ratio_height > ratio_width )
        {
            d_bbox.x      = (d_bbox.x / ratio_width);
            d_bbox.y      = ((d_bbox.y - (tensor_input_height - ratio_width * image.rows) / 2) / ratio_width);
            d_bbox.width  = (d_bbox.width / ratio_width);
            d_bbox.height = (d_bbox.height / ratio_width);
        }
        else
        {
            d_bbox.x      = ((d_bbox.x - (tensor_input_width - ratio_height * image.cols) / 2) / ratio_height);
            d_bbox.y      = (d_bbox.y / ratio_height);
            d_bbox.width  = (d_bbox.width / ratio_height);
            d_bbox.height = (d_bbox.height / ratio_height);
        }

        if ( (d_bbox.width > false_detect_w) && (d_bbox.height > false_detect_h) ) { continue; } // false detect check

        cv::Point  top_left(d_bbox.x, d_bbox.y);
        cv::Point  bottom_right((d_bbox.x + d_bbox.width), (d_bbox.y + d_bbox.height));
        cv::Scalar color_green(0, 255, 0);

        cv::rectangle(image, top_left, bottom_right, color_green, /*line width*/2.0);
    }

    return ( image );
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
