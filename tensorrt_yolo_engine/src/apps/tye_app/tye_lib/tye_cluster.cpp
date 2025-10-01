//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "tye_cluster.h"

//-- constructor(s) --------------------------------------------------------------------------------------------------------

tye_cluster::tye_cluster( std::vector<uint32_t> gpus, std::string engine_path, uint32_t num_engines, uint32_t fft_size,
                          float score_threshold, float nms_threshold, float power_offset, float pixel_min_val,
                          float pixel_max_val, uint32_t false_detect_w, uint32_t false_detect_h,
                          void (*p_notify_cb)( tye_types::detections_tuple dt, void *p_data ), void *p_notify_cb_data,
                          bool test_debug )
{
    // initialize
    this_gpus             = gpus;
    this_engine_path      = engine_path;
    this_num_engines      = num_engines;
    this_fft_size         = fft_size;
    this_score_threshold  = score_threshold;
    this_nms_threshold    = nms_threshold;
    this_power_offset     = power_offset;
    this_pixel_min_val    = pixel_min_val;
    this_pixel_max_val    = pixel_max_val;
    this_false_detect_w   = false_detect_w;
    this_false_detect_h   = false_detect_h;
    this_p_notify_cb      = p_notify_cb;
    this_p_notify_cb_data = p_notify_cb_data;
    this_test_debug       = test_debug;
    this_next_processor   = 0;
    this_p_notifier       = nullptr;

    this_processors.clear();

    return;    
}

//-- destructor ------------------------------------------------------------------------------------------------------------

tye_cluster::~tye_cluster( void ) { return; }

//-- public methods --------------------------------------------------------------------------------------------------------

// start the egnine cluster [1 or more instances of the same engine across 1 or more GPUs]
bool tye_cluster::start( uint32_t wait_ready_timeout_sec )
{
    tye_processor *p_processor = nullptr;
    uint32_t       num_gpus    = (uint32_t)this_gpus.size();
    bool           ok          = false;

    // create and start the notifier
    this_p_notifier = new tye_notifier(this_p_notify_cb, this_p_notify_cb_data);
    if ( this_p_notifier == nullptr ) { goto FAILED; }

    ok = this_p_notifier->start();
    if ( ! ok ) { goto FAILED_FREE_NOTIFIER; }

    // start processors
    for ( uint32_t gpu_idx = 0; gpu_idx < num_gpus; gpu_idx++ )
    {
        uint32_t gpu = this_gpus.at(gpu_idx);

        for ( uint32_t engine_id = 0; engine_id < this_num_engines; engine_id++ )
        {
            p_processor = new tye_processor(gpu, this_engine_path, engine_id, this_fft_size, this_score_threshold,
                                            this_nms_threshold, this_power_offset, this_pixel_min_val, this_pixel_max_val,
                                            this_false_detect_w, this_false_detect_h, this_p_notifier, this_test_debug);
            if ( p_processor == nullptr ) { goto FAILED_SHUTDOWN_NOTIFIER_AND_PROCESSORS; }

            ok = p_processor->start();
            if ( ! ok ) { goto FAILED; }

            this_processors.push_back(p_processor);
        }
    }

    // wait for processors to be ready
    ok = this->wait_processors_ready(wait_ready_timeout_sec);
    if ( ! ok ) { goto FAILED; }

    return ( true );

FAILED_SHUTDOWN_NOTIFIER_AND_PROCESSORS:
    this_p_notifier->shutdown();
    this->shutdown_processors();

FAILED_FREE_NOTIFIER:
    delete this_p_notifier;
    this_p_notifier = nullptr;

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// shutdown the engine cluster
void tye_cluster::shutdown( void )
{
    this->shutdown_processors();

    if ( this_p_notifier != nullptr )
    {
        if ( this_p_notifier->is_running() ) { this_p_notifier->shutdown(); }

        delete this_p_notifier;
        this_p_notifier = nullptr;
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// update some of the detection-related parameters
void tye_cluster::update_params( float power_offset, float pixel_min_val, float pixel_max_val, uint32_t false_detect_w,
                                 uint32_t false_detect_h )
{
    this_power_offset   = power_offset;
    this_pixel_min_val  = pixel_min_val;
    this_pixel_max_val  = pixel_max_val;
    this_false_detect_w = false_detect_w;
    this_false_detect_h = false_detect_h;

    for ( uint32_t i = 0; i < this_processors.size(); i++ )
    {
        this_processors.at(i)->update_params(this_power_offset, this_pixel_min_val, this_pixel_max_val,
                                             this_false_detect_w, this_false_detect_h);
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// process samples
void tye_cluster::process( tye_buffer *p_samples )
{
    this_processors.at(this_next_processor)->process(p_samples);

    this_next_processor++;
    if ( this_next_processor == this_processors.size() ) { this_next_processor = 0; }

    return;
}

//-- private methods -------------------------------------------------------------------------------------------------------

// wait for all processors to be ready
bool tye_cluster::wait_processors_ready( uint32_t timeout_sec )
{
    time_t ready_timeout = (time(nullptr) + (time_t)timeout_sec);

    while ( true )
    {
        if ( time(nullptr) > ready_timeout ) { goto FAILED; }

        uint32_t num_processors = (uint32_t)this_processors.size();
        uint32_t ready_cnt      = 0;

        for ( uint32_t i = 0; i < num_processors; i++ )
        {
            bool ready = this_processors.at(i)->is_ready();
            if ( ready ) { ready_cnt++; }
        }

        if ( ready_cnt == num_processors ) { break; }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    return ( true );

FAILED:
    this->shutdown_processors();

    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// shutdown and destroy all engine cluster processors
void tye_cluster::shutdown_processors( void )
{
    uint32_t num_processors = (uint32_t)this_processors.size();

    for ( uint32_t i = 0; i < num_processors; i++ )
    {
        if ( this_processors.at(i)->is_running() ) { this_processors.at(i)->shutdown(); }
        delete this_processors.at(i);
    }

    this_processors.clear();

    return;
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
