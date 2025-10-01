#ifndef INCLUDE_SP_CONFIG_H
#define INCLUDE_SP_CONFIG_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_app_includes.h"
#include "config.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class sp_config : public config
{
public: //==================================================================================================================

    // constructor(s) / destructor
    sp_config( std::vector<uint32_t> gpus, std::string engine_path, uint32_t engines_per_gpu, uint16_t ad_port,
               uint16_t cnc_port, uint16_t retune_port, std::string database_creds, std::string database_ip,
               uint16_t database_port, bool database_ff, uint32_t fft_size, uint64_t sample_rate_hz, uint64_t center_freq_hz,
               int32_t atten_db, double ref_level, float score_threshold, float nms_threshold, float power_offset,
               float pixel_min_val, float pixel_max_val, uint32_t false_detect_w, uint32_t false_detect_h, bool
               boxes_plot_on, bool history_plot_on );

   ~sp_config( void );
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_SP_CONFIG_H
