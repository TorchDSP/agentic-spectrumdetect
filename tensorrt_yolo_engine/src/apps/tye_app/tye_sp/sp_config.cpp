//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "sp_config.h"

//-- constructor(s) --------------------------------------------------------------------------------------------------------

sp_config::sp_config( std::vector<uint32_t> gpus, std::string engine_path, uint32_t engines_per_gpu, uint16_t ad_port,
                      uint16_t cnc_port, uint16_t retune_port, std::string database_creds, std::string database_ip,
                      uint16_t database_port, bool database_off, uint32_t fft_size, uint64_t sample_rate_hz,
                      uint64_t center_freq_hz, int32_t atten_db, double ref_level, float score_threshold,
                      float nms_threshold, float power_offset, float pixel_min_val, float pixel_max_val,
                      uint32_t false_detect_w, uint32_t false_detect_h, bool boxes_plot_on, bool history_plot_on )
  : config( true, std::string(""), gpus, engine_path, engines_per_gpu, std::string(""), ad_port, cnc_port, retune_port,
            database_creds, database_ip, database_port, database_off, fft_size, sample_rate_hz, center_freq_hz, atten_db,
            ref_level, score_threshold, nms_threshold, power_offset, pixel_min_val, pixel_max_val, false_detect_w,
            false_detect_h, boxes_plot_on, history_plot_on )
{
    return;
}

//-- destructor ------------------------------------------------------------------------------------------------------------

sp_config::~sp_config( void ) { return; }

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
