#ifndef INCLUDE_TYE_TYPES_H
#define INCLUDE_TYE_TYPES_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_includes.h"
#include "tye_buffer.h"
#include "tye_spectrogram.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

namespace tye_types
{
    typedef struct
    {
        float    score;
        float    rssi;
        cv::Rect bbox;

    } detection;

    typedef struct
    {
        std::string source_name;
        uint64_t    group_id;
        uint64_t    sequ_num;

        struct
        {
            uint64_t ns_since_epoch;
            uint64_t sample_rate_hz;
            uint32_t bandwidth_hz;
            uint64_t center_freq_hz;
            int32_t  atten_db;
            double   ref_level;

        } radio;

        struct
        {
            double preprocessing_usec;
            double spectrogram_usec;
            double inference_usec;
            double get_detections_usec;
            double nms_boxes_usec;
            double rssi_calc_usec;
            double all_proc_usec;

        } timing;

    } detections_aux;

    typedef std::vector<detection>                                                detections_vec;
    typedef std::tuple<detections_vec, detections_aux, cv::Mat, tye_spectrogram*> detections_tuple;
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_TYE_TYPES_H
