//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "fp_config.h"

//-- constructor(s) --------------------------------------------------------------------------------------------------------

fp_config::fp_config( std::vector<uint32_t> gpus, std::string engine_path, uint32_t engines_per_gpu,
                      std::string process_path, std::string output_path, uint16_t ad_port, uint16_t cnc_port,
                      std::string database_creds, std::string database_ip, uint16_t database_port, bool database_off,
                      uint32_t fft_size, float score_threshold, float nms_threshold, float power_offset,
                      float pixel_min_val, float pixel_max_val, uint32_t false_detect_w, uint32_t false_detect_h )
  : config( false, process_path, gpus, engine_path, engines_per_gpu, output_path, ad_port, cnc_port, 0, database_creds,
            database_ip, database_port, database_off, fft_size, 0, 0, 0, 0.0, score_threshold, nms_threshold, power_offset,
            pixel_min_val, pixel_max_val, false_detect_w, false_detect_h, false, false)
{
    return;
}

//-- destructor ------------------------------------------------------------------------------------------------------------

fp_config::~fp_config( void ) { return; }

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
