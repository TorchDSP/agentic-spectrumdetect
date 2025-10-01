#ifndef INCLUDE_SH_SM_RADIO_H
#define INCLUDE_SH_SM_RADIO_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_app_includes.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class sh_sm_radio
{
public: //==================================================================================================================

    // constructor(s) / destructor
    sh_sm_radio( void );
   ~sh_sm_radio( void );

    // public methods
    bool        connect_usb( void );
    bool        connect_net( std::string ipaddr, uint16_t port );
    bool        configure( uint64_t sample_rate_hz, uint64_t center_freq_hz, int32_t atten_db, double ref_level );
    bool        recv_samples( void *p_buffer, uint32_t buffer_len, int64_t *p_ns_since_epoch, bool *p_samples_dropped );
    void        disconnect( void );
    std::string get_device_name( void );
    uint64_t    get_sample_rate_hz( void );
    double      get_sample_rate_mhz( void );
    uint64_t    get_sample_rate_max_hz( void );
    uint64_t    get_bandwidth_hz( void );
    double      get_bandwidth_mhz( void );
    uint64_t    get_center_freq_hz( void );
    double      get_center_freq_mhz( void );
    uint64_t    get_center_freq_min_hz( void );
    uint64_t    get_center_freq_max_hz( void );
    bool        atten_is_auto( void );
    int32_t     get_atten_db( void );
    double      get_ref_level( void );

private: //=================================================================================================================

    // private constants
    const std::string NAME = std::string("SH_SM_RADIO");

    // private variables
    int32_t     this_device_id;
    int32_t     this_device_handle;
    int32_t     this_device_type;
    std::string this_device_name;
    uint64_t    this_sample_rate_hz;
    double      this_sample_rate_mhz;
    uint64_t    this_sample_rate_max_hz;
    uint64_t    this_bandwidth_hz;
    double      this_bandwidth_mhz;
    uint64_t    this_center_freq_hz;
    double      this_center_freq_mhz;
    uint64_t    this_center_freq_min_hz;
    uint64_t    this_center_freq_max_hz;
    int32_t     this_atten_value;
    int32_t     this_atten_db;
    double      this_ref_level;
    bool        this_connected;

    std::vector<std::string> this_device_names;
    std::vector<uint64_t>    this_sample_rates;
    std::vector<int32_t>     this_atten_values;
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_SH_SM_RADIO_H
