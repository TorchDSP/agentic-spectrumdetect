#ifndef INCLUDE_TYE_H
#define INCLUDE_TYE_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_notifier.h"
#include "tye_processor.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class tye_cluster
{
public: //==================================================================================================================

    // constructor(s) / destructor
    tye_cluster( std::vector<uint32_t> gpus, std::string engine_path, uint32_t num_engines, uint32_t fft_size,
                 float power_offset, float score_threshold, float nms_threshold, float pixel_min_val,
                 float pixel_max_val, uint32_t false_detect_w, uint32_t false_detect_h,
                 void (*p_notify_cb)( tye_types::detections_tuple dt, void *p_data ), void *p_notify_cb_data,
                 bool test_debug = false );
   ~tye_cluster( void );

    // public methods
    bool start( uint32_t wait_ready_timeout_sec = 30 );
    void shutdown( void );
    void update_params( float power_offset, float pixel_min_val, float pixel_max_val, uint32_t false_detect_w,
                        uint32_t false_detect_h );
    void process( tye_buffer *p_samples );

private: //=================================================================================================================

    // private constants
    const std::string NAME = std::string("TYE_CLUSTER");

    // private variables
    void (*this_p_notify_cb)( tye_types::detections_tuple dt, void *p_data );
    void  *this_p_notify_cb_data;

    std::string this_engine_path;
    uint32_t    this_num_engines;
    uint32_t    this_fft_size;
    float       this_power_offset;
    float       this_score_threshold;
    float       this_nms_threshold;
    float       this_pixel_min_val;
    float       this_pixel_max_val;
    uint32_t    this_false_detect_w;
    uint32_t    this_false_detect_h;
    bool        this_test_debug;
    uint32_t    this_next_processor;

    std::vector<uint32_t>        this_gpus;
    tye_notifier                *this_p_notifier;
    std::vector<tye_processor *> this_processors;

    // private methods
    bool wait_processors_ready( uint32_t timeout_sec );
    void shutdown_processors( void );
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_TYE_CLUSTER_H
