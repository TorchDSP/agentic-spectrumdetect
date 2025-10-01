#ifndef INCLUDE_CONFIG_H
#define INCLUDE_CONFIG_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_app_includes.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class config
{
public: //==================================================================================================================

    // constructor(s) / destructor
    config( bool process_stream, std::string process_path, std::vector<uint32_t> gpus, std::string engine_path,
            uint32_t engines_per_gpu, std::string output_path, uint16_t ad_port, uint16_t cnc_port, uint16_t retune_port,
            std::string database_creds, std::string database_ip, uint16_t database_port, bool database_off,
            uint32_t fft_size, uint64_t sample_rate_hz, uint64_t center_freq_hz, int32_t atten_db, double ref_level,
            float score_threshold, float nms_threshold, float power_offset, float pixel_min_val, float pixel_max_val,
            uint32_t false_detect_w, uint32_t false_detect_h, bool boxes_plot_on, bool history_plot_on );

    virtual ~config( void );

    // public methods
    void                  set_process_stream( bool on );
    bool                  process_stream( void );
    void                  set_process_path( std::string path );
    std::string           process_path( void );
    void                  set_gpus( std::vector<uint32_t> gpus );
    std::vector<uint32_t> gpus( void );
    void                  set_engine_path( std::string path );
    std::string           engine_path( void );
    void                  set_engines_per_gpu( uint32_t num );
    uint32_t              engines_per_gpu( void );
    void                  set_output_path( std::string path );
    std::string           output_path( void );
    void                  set_ad_port( uint16_t port );
    uint16_t              ad_port( void );
    void                  set_cnc_port( uint16_t port );
    uint16_t              cnc_port( void );
    void                  set_retune_port( uint16_t port );
    uint16_t              retune_port( void );
    void                  set_database_creds( std::string creds );
    std::string           database_creds( void );
    void                  set_database_ip( std::string ip );
    std::string           database_ip( void );
    void                  set_database_port( uint16_t port );
    uint16_t              database_port( void );
    void                  set_database_off( bool off );
    bool                  database_off( void );
    void                  set_fft_size( uint32_t size );
    uint32_t              fft_size( void );
    void                  set_sample_rate_hz( uint64_t rate_hz );
    uint64_t              sample_rate_hz( void );
    void                  set_center_freq_hz( uint64_t freq_hz );
    uint64_t              center_freq_hz( void );
    void                  set_atten_db( int32_t atten_db );
    int32_t               atten_db( void );
    void                  set_ref_level( double level );
    double                ref_level( void );
    void                  set_score_threshold( float threshold );
    float                 score_threshold( void );
    void                  set_nms_threshold( float threshold );
    float                 nms_threshold( void );
    void                  set_power_offset( float offset );
    float                 power_offset( void );
    void                  set_pixel_min_val( float val );
    float                 pixel_min_val( void );
    void                  set_pixel_max_val( float val );
    float                 pixel_max_val( void );
    void                  set_false_detect_w( uint32_t width );
    uint32_t              false_detect_w( void );
    void                  set_false_detect_h( uint32_t height );
    uint32_t              false_detect_h( void );
    void                  set_boxes_plot_on( bool on );
    bool                  boxes_plot_on( void );
    void                  set_history_plot_on( bool on );
    bool                  history_plot_on( void );

protected: //===============================================================================================================

    // protected variables
    bool                  this_process_stream;
    std::string           this_process_path;
    std::vector<uint32_t> this_gpus;
    std::string           this_engine_path;
    uint32_t              this_engines_per_gpu;
    std::string           this_output_path;
    uint16_t              this_ad_port;
    uint16_t              this_cnc_port;
    uint16_t              this_retune_port;
    std::string           this_database_creds;
    std::string           this_database_ip;
    uint16_t              this_database_port;
    bool                  this_database_off;
    uint32_t              this_fft_size;
    uint64_t              this_sample_rate_hz;
    uint64_t              this_center_freq_hz;
    int32_t               this_atten_db;
    double                this_ref_level;
    float                 this_score_threshold;
    float                 this_nms_threshold;
    float                 this_power_offset;
    float                 this_pixel_min_val;
    float                 this_pixel_max_val;
    uint32_t              this_false_detect_w;
    uint32_t              this_false_detect_h;
    bool                  this_boxes_plot_on;
    bool                  this_history_plot_on;
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_CONFIG_H
