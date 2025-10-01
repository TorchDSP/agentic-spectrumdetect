#ifndef INCLUDE_TYE_PROCESSOR_H
#define INCLUDE_TYE_PROCESSOR_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_buffer.h"
#include "tye_spectrogram_pool.h"
#include "tye_ops.h"
#include "tye_notifier.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class tye_processor
{
public: //==================================================================================================================

    // constructor(s) / destructor
    tye_processor( uint32_t gpu, std::string engine_path, uint32_t engine_id, uint32_t fft_size, float score_threshold,
                   float nms_threshold, float power_offset, float pixel_min_val, float pixel_max_val, uint32_t false_detect_w,
                   uint32_t false_detect_h, tye_notifier *p_notifier, bool test_debug );
   ~tye_processor( void );

    // public methods
    bool start( void );
    void shutdown( void );
    bool is_running( void );
    bool is_ready( void );
    void update_params( float power_offset, float pixel_min_val, float pixel_max_val, uint32_t false_detect_w,
                        uint32_t false_detect_h );
    void process( tye_buffer *p_samples );

private: //=================================================================================================================

    // private constants
    const std::string NAME = std::string("TYE_PROCESSOR");

    // private variables
    std::thread *this_p_mgr_thread;
    std::mutex   this_mutex;
    uint32_t     this_gpu;
    std::string  this_engine_path;
    uint32_t     this_engine_id;
    uint32_t     this_fft_size;
    float        this_score_threshold;
    float        this_nms_threshold;
    float        this_power_offset;
    float        this_pixel_min_val;
    float        this_pixel_max_val;
    uint32_t     this_false_detect_w;
    uint32_t     this_false_detect_h;
    bool         this_test_debug;
    bool         this_running;
    bool         this_ready;
    bool         this_exit;

    tye_spectrogram_pool     *this_p_spectrogram_pool;
    tye_ops                  *this_p_engine_ops;
    tye_notifier             *this_p_notifier;
    std::vector<tye_buffer *> this_samples;

    // private methods
    void warmup( void );
    bool samples_to_process( void );
    void process_samples( void );
    void mgr_thread( void );
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_TYE_PROCESSOR_H
