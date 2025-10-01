//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "tye_processor.h"

//-- tensor-rt inference logger --------------------------------------------------------------------------------------------

class Logger : public nvinfer1::ILogger
{
    void log( Severity severity, const char* msg ) noexcept override
    {
        if ( severity <= Severity::kWARNING ) { std::cout << msg << std::endl; }
    }

} tye_processor_logger;

//-- constructor(s) --------------------------------------------------------------------------------------------------------

tye_processor::tye_processor( uint32_t gpu, std::string engine_path, uint32_t engine_id, uint32_t fft_size,
                              float score_threshold, float nms_threshold, float power_offset, float pixel_min_val,
                              float pixel_max_val, uint32_t false_detect_w, uint32_t false_detect_h,
                              tye_notifier *p_notifier, bool test_debug )
{
    // initialize
    this_p_mgr_thread       = nullptr;
    this_p_spectrogram_pool = nullptr;
    this_p_engine_ops       = nullptr;
    this_p_notifier         = p_notifier;
    this_gpu                = gpu;
    this_engine_path        = engine_path;
    this_engine_id          = engine_id;
    this_fft_size           = fft_size;
    this_score_threshold    = score_threshold;
    this_nms_threshold      = nms_threshold;
    this_power_offset       = power_offset;
    this_pixel_min_val      = pixel_min_val;
    this_pixel_max_val      = pixel_max_val;
    this_false_detect_w     = false_detect_w;
    this_false_detect_h     = false_detect_h;
    this_test_debug         = test_debug;
    this_running            = false;
    this_ready              = false;
    this_exit               = false;

    this_samples.clear();

    return;    
}

//-- destructor ------------------------------------------------------------------------------------------------------------

tye_processor::~tye_processor( void ) { return; }

//-- public methods --------------------------------------------------------------------------------------------------------

// start the processor
bool tye_processor::start( void )
{
    this_p_mgr_thread = new std::thread(&tye_processor::mgr_thread, this);
    if ( this_p_mgr_thread == nullptr ) { goto FAILED; }

    return ( true );

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// shutdown the processor
void tye_processor::shutdown( void )
{
    if ( this_running )
    {
        this_exit = true;
        this_p_mgr_thread->join();
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// check if the processor is running/ready
bool tye_processor::is_running( void ) { return ( this_running ); }
bool tye_processor::is_ready( void )   { return ( this_ready   ); }

//--------------------------------------------------------------------------------------------------------------------------

// update some of the detection-related parameters
void tye_processor::update_params( float power_offset, float pixel_min_val, float pixel_max_val, uint32_t false_detect_w,
                                   uint32_t false_detect_h )
{
    this_power_offset   = power_offset;
    this_pixel_min_val  = pixel_min_val;
    this_pixel_max_val  = pixel_max_val;
    this_false_detect_w = false_detect_w;
    this_false_detect_h = false_detect_h;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// enqueue samples for processing
void tye_processor::process( tye_buffer *p_samples )
{
    this_mutex.lock();
    this_samples.push_back(p_samples);
    this_mutex.unlock();

    return;
}

//-- private methods -------------------------------------------------------------------------------------------------------

// warm-up inference, GPUs, etc
void tye_processor::warmup( void )
{
    uint32_t warmup_cnt  = 50;
    uint32_t samples_len = (this_fft_size * this_fft_size * sizeof(float) * 2);
    uint8_t *p_samples   = (uint8_t *)malloc(samples_len);

    if ( p_samples != nullptr )
    {
        for ( uint32_t i = 0; i < warmup_cnt; i++ )
        {
            cv::Mat                           image_host = cv::Mat();
            std::vector<tye_types::detection> detections = std::vector<tye_types::detection>();
            double                            sg_usec    = 0.0;
            double                            nms_usec   = 0.0;

            this_p_engine_ops->preprocess_samples(p_samples, samples_len, this_pixel_min_val, this_pixel_max_val,
                                                  image_host, nullptr, &sg_usec);
            this_p_engine_ops->run_inference();
            this_p_engine_ops->get_detections(this_score_threshold, this_nms_threshold, &nms_usec, detections);
        }

        free(p_samples);
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// check if there are samples to process
bool tye_processor::samples_to_process( void )
{
    bool have_samples_to_process = false;

    this_mutex.lock();
    if ( this_samples.size() > 0 ) { have_samples_to_process = true; }
    this_mutex.unlock();

    return ( have_samples_to_process );
}

//--------------------------------------------------------------------------------------------------------------------------

// process samples
void tye_processor::process_samples( void )
{
    tye_buffer *p_samples = nullptr;

    // attempt to dequeue samples for processing
    this_mutex.lock();

    if ( this_samples.size() > 0 )
    {
        p_samples = this_samples.at(0);
        this_samples.erase(this_samples.begin());
    }
    this_mutex.unlock();

    // if there are samples to process
    if ( p_samples != nullptr )
    {
        // process samples
        cv::Mat                           image_host     = cv::Mat();
        std::vector<tye_types::detection> detections     = std::vector<tye_types::detection>();
        std::string                       source_name    = p_samples->get_source_name();
        uint64_t                          group_id       = p_samples->get_group_id();
        uint64_t                          sequ_num       = p_samples->get_sequ_num();
        uint64_t                          samples_ns     = p_samples->get_samples_ns_since_epoch();
        uint64_t                          sample_rate_hz = p_samples->get_sample_rate_hz();
        uint64_t                          bandwidth_hz   = p_samples->get_bandwidth_hz();
        uint64_t                          center_freq_hz = p_samples->get_center_freq_hz();
        int32_t                           atten_db       = p_samples->get_atten_db();
        double                            ref_level      = p_samples->get_ref_level();
        double                            sg_usec        = 0.0;
        double                            nms_usec       = 0.0;

        tye_spectrogram *p_sg = nullptr;
        while ( true ) // should never get stuck in this loop
        {
            p_sg = this_p_spectrogram_pool->get();
            if ( p_sg != nullptr ) { break; }
        }

        auto   proc_start = std::chrono::system_clock::now();
        double prepr_usec = this_p_engine_ops->preprocess_samples(p_samples->get(), p_samples->len(), this_pixel_min_val,
                                                                  this_pixel_max_val, image_host, (float *)p_sg->buffer(),
                                                                  &sg_usec);
        p_samples->release();

        double infer_usec = this_p_engine_ops->run_inference();
        double getdt_usec = this_p_engine_ops->get_detections(this_score_threshold, this_nms_threshold, &nms_usec,
                                                              detections);
        double rssi_usec  = this_p_engine_ops->calc_detections_rssi((float *)p_sg->buffer(), this_fft_size, this_fft_size,
                                                                    detections, this_power_offset);
        auto   proc_end   = std::chrono::system_clock::now();
        double proc_usec  = (double)std::chrono::duration_cast<std::chrono::microseconds>(proc_end - proc_start).count();

        // store auxiliary information that is returned with detections
        tye_types::detections_aux detections_aux = {};

        detections_aux.source_name                = source_name;
        detections_aux.group_id                   = group_id;
        detections_aux.sequ_num                   = sequ_num;
        detections_aux.radio.ns_since_epoch       = samples_ns;
        detections_aux.radio.sample_rate_hz       = sample_rate_hz;
        detections_aux.radio.bandwidth_hz         = bandwidth_hz;
        detections_aux.radio.center_freq_hz       = center_freq_hz;
        detections_aux.radio.atten_db             = atten_db;
        detections_aux.radio.ref_level            = ref_level;
        detections_aux.timing.preprocessing_usec  = prepr_usec;
        detections_aux.timing.spectrogram_usec    = sg_usec;
        detections_aux.timing.inference_usec      = infer_usec;
        detections_aux.timing.get_detections_usec = getdt_usec;
        detections_aux.timing.nms_boxes_usec      = nms_usec;
        detections_aux.timing.rssi_calc_usec      = rssi_usec;
        detections_aux.timing.all_proc_usec       = proc_usec;

        // get image with detection bounding boxes
        cv::Mat image_with_detections = this_p_engine_ops->get_image_with_detections(image_host, this_false_detect_w,
                                                                                     this_false_detect_h, detections);
        // enqueue detections for notification
        this_p_notifier->enqueue(std::make_tuple(detections, detections_aux, image_with_detections, p_sg));
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// processor manager [main] instance
void tye_processor::mgr_thread( void )
{
    this_running = true;

    // create engine operations
    this_p_engine_ops = new tye_ops(this_gpu, this_engine_path, this_engine_id, this_fft_size);
    if ( this_p_engine_ops == nullptr )
    {
        throw std::runtime_error(tye_processor::NAME + " [EXCEPTION] CREATE ENGINE OPERATIONS");
    }

    // create the spectrogram buffer pool
    this_p_spectrogram_pool = new tye_spectrogram_pool(this_gpu, this_fft_size, 32/*spectrograms*/);
    if ( this_p_spectrogram_pool == nullptr )
    {
        throw std::runtime_error(tye_processor::NAME + " [EXCEPTION] CREATE SPECTROGRAM BUFFER POOL [1]");
    }

    bool ok = this_p_spectrogram_pool->create();
    if ( ! ok )
    {
        throw std::runtime_error(tye_processor::NAME + " [EXCEPTION] CREATE SPECTROGRAM BUFFER POOL [2]");
    }

    // is ready after warmup finishes
    this->warmup();
    this_ready = true;

    while ( true )
    {
        // exit requested ??
        if ( this_exit ) { break; }

        if ( this->samples_to_process() ) { this->process_samples(); }
        std::this_thread::yield();
    }

    // clean up
    delete this_p_spectrogram_pool;
    delete this_p_engine_ops;

    this_running = false;
    this_ready   = false;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
