//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "config.h"

//-- constructor(s) --------------------------------------------------------------------------------------------------------

config::config( bool process_stream, std::string process_path, std::vector<uint32_t> gpus, std::string engine_path,
                uint32_t engines_per_gpu, std::string output_path, uint16_t ad_port, uint16_t cnc_port,
                uint16_t retune_port, std::string database_creds, std::string database_ip, uint16_t database_port,
                bool database_off, uint32_t fft_size, uint64_t sample_rate_hz, uint64_t center_freq_hz, int32_t atten_db,
                double ref_level, float score_threshold, float nms_threshold, float power_offset, float pixel_min_val,
                float pixel_max_val, uint32_t false_detect_w, uint32_t false_detect_h, bool boxes_plot_on,
                bool history_plot_on )
{
    // initialize
    this_process_stream  = process_stream;
    this_process_path    = process_path;
    this_gpus            = gpus;
    this_engine_path     = engine_path;
    this_engines_per_gpu = engines_per_gpu;
    this_output_path     = output_path;
    this_ad_port         = ad_port;
    this_cnc_port        = cnc_port;
    this_retune_port     = retune_port;
    this_database_creds  = database_creds;
    this_database_ip     = database_ip;
    this_database_port   = database_port;
    this_database_off    = database_off;
    this_fft_size        = fft_size;
    this_sample_rate_hz  = sample_rate_hz;
    this_center_freq_hz  = center_freq_hz;
    this_atten_db        = atten_db;
    this_ref_level       = ref_level;
    this_score_threshold = score_threshold;
    this_nms_threshold   = nms_threshold;
    this_power_offset    = power_offset;
    this_pixel_min_val   = pixel_min_val;
    this_pixel_max_val   = pixel_max_val;
    this_false_detect_w  = false_detect_w;
    this_false_detect_h  = false_detect_h;
    this_boxes_plot_on   = boxes_plot_on;
    this_history_plot_on = history_plot_on;

    return;
}

//-- destructor ------------------------------------------------------------------------------------------------------------

config::~config( void ) { return; }

//-- public methods --------------------------------------------------------------------------------------------------------

// setters
void config::set_process_stream( bool on )           { this_process_stream    = on;        }
void config::set_process_path( std::string path )    { this_process_path      = path;      }
void config::set_gpus( std::vector<uint32_t> gpus )  { this_gpus              = gpus;      }
void config::set_engine_path( std::string path )     { this_engine_path       = path;      }
void config::set_engines_per_gpu( uint32_t num )     { this_engines_per_gpu   = num;       }
void config::set_output_path( std::string path )     { this_output_path       = path;      }
void config::set_ad_port( uint16_t port )            { this_ad_port           = port;      }
void config::set_cnc_port( uint16_t port )           { this_cnc_port          = port;      }
void config::set_retune_port( uint16_t port )        { this_retune_port       = port;      }
void config::set_database_creds( std::string creds ) { this_database_creds    = creds;     }
void config::set_database_ip( std::string ip )       { this_database_ip       = ip;        }
void config::set_database_port( uint16_t port )      { this_database_port     = port;      }
void config::set_database_off( bool off )            { this_database_off      = off;       }
void config::set_fft_size( uint32_t size )           { this_fft_size          = size;      }
void config::set_sample_rate_hz( uint64_t rate_hz )  { this_sample_rate_hz    = rate_hz;   }
void config::set_center_freq_hz( uint64_t freq_hz )  { this_center_freq_hz    = freq_hz;   }
void config::set_atten_db( int32_t atten_db )        { this_atten_db          = atten_db;  }
void config::set_ref_level( double level )           { this_ref_level         = level;     }
void config::set_score_threshold( float threshold )  { this_score_threshold   = threshold; }
void config::set_nms_threshold( float threshold )    { this_nms_threshold     = threshold; }
void config::set_power_offset( float offset )        { this_power_offset      = offset;    }
void config::set_pixel_min_val( float val )          { this_pixel_min_val     = val;       }
void config::set_pixel_max_val( float val )          { this_pixel_max_val     = val;       }
void config::set_false_detect_w( uint32_t width )    { this_false_detect_w    = width;     }
void config::set_false_detect_h( uint32_t height )   { this_false_detect_h    = height;    }
void config::set_boxes_plot_on( bool on )            { this_boxes_plot_on     = on;        }
void config::set_history_plot_on( bool on )          { this_history_plot_on   = on;        }

//--------------------------------------------------------------------------------------------------------------------------

// getters
bool                  config::process_stream( void )    { return ( this_process_stream  ); }
std::string           config::process_path( void )      { return ( this_process_path    ); }
std::vector<uint32_t> config::gpus( void )              { return ( this_gpus            ); }
std::string           config::engine_path( void )       { return ( this_engine_path     ); }
uint32_t              config::engines_per_gpu( void )   { return ( this_engines_per_gpu ); }
std::string           config::output_path( void )       { return ( this_output_path     ); }
uint16_t              config::ad_port( void )           { return ( this_ad_port         ); }
uint16_t              config::cnc_port( void )          { return ( this_cnc_port        ); }
uint16_t              config::retune_port( void )       { return ( this_retune_port     ); }
std::string           config::database_creds( void )    { return ( this_database_creds  ); }
std::string           config::database_ip( void )       { return ( this_database_ip     ); }
uint16_t              config::database_port( void )     { return ( this_database_port   ); }
bool                  config::database_off( void )      { return ( this_database_off    ); }
uint32_t              config::fft_size( void )          { return ( this_fft_size        ); }
uint64_t              config::sample_rate_hz( void )    { return ( this_sample_rate_hz  ); }
uint64_t              config::center_freq_hz( void )    { return ( this_center_freq_hz  ); }
int32_t               config::atten_db( void )          { return ( this_atten_db        ); }
double                config::ref_level( void )         { return ( this_ref_level       ); }
float                 config::score_threshold( void )   { return ( this_score_threshold ); }
float                 config::nms_threshold( void )     { return ( this_nms_threshold   ); }
float                 config::power_offset( void )      { return ( this_power_offset    ); }
float                 config::pixel_min_val( void )     { return ( this_pixel_min_val   ); }
float                 config::pixel_max_val( void )     { return ( this_pixel_max_val   ); }
uint32_t              config::false_detect_w( void )    { return ( this_false_detect_w  ); }
uint32_t              config::false_detect_h( void )    { return ( this_false_detect_h  ); }
bool                  config::boxes_plot_on( void )     { return ( this_boxes_plot_on   ); }
bool                  config::history_plot_on( void )   { return ( this_history_plot_on ); }

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
