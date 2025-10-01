#ifndef INCLUDE_FP_CONFIG_H
#define INCLUDE_FP_CONFIG_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_app_includes.h"
#include "config.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class fp_config : public config
{
public: //==================================================================================================================

    // constructor(s) / destructor
    fp_config( std::vector<uint32_t> gpus, std::string engine_path, uint32_t engines_per_gpu,
               std::string process_path, std::string output_path, uint16_t ad_port, uint16_t cnc_port,
               std::string database_creds, std::string database_ip, uint16_t database_port, bool database_off,
               uint32_t fft_size, float score_threshold, float nms_threshold, float power_offset, float pixel_min_val,
               float pixel_max_val, uint32_t false_detect_w, uint32_t false_detect_h );

   ~fp_config( void );
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_FP_CONFIG_H
