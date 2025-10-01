//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "sh_sm_radio.h"

//-- constructor(s) --------------------------------------------------------------------------------------------------------

sh_sm_radio::sh_sm_radio( void )
{
    // initialize
    this_device_id          = -1;
    this_device_handle      = -1;
    this_device_type        = -1;
    this_device_name        = std::string("");
    this_sample_rate_hz     = 0;
    this_sample_rate_mhz    = 0.0;
    this_sample_rate_max_hz = 50000000;/*50 MHz*/
    this_bandwidth_hz       = 0;
    this_bandwidth_mhz      = 0.0;
    this_center_freq_hz     = 0;
    this_center_freq_min_hz = 0;
    this_center_freq_max_hz = 0;
    this_center_freq_mhz    = 0.0;
    this_atten_value        = -1;
    this_atten_db           = -1;
    this_ref_level          = 0.0;
    this_connected          = false;

    this_device_names.clear();
    this_sample_rates.clear();
    this_atten_values.clear();

    // build list of device names
    this_device_names.push_back(std::string("SM200A"));
    this_device_names.push_back(std::string("SM200B"));
    this_device_names.push_back(std::string("SM200C"));
    this_device_names.push_back(std::string("SM435B"));
    this_device_names.push_back(std::string("SM435C"));

    // build list of sample rates
    for ( uint32_t i = 1; i <= 128; i *= 2 ) { this_sample_rates.push_back(this_sample_rate_max_hz / i); }

    // build list of attenuation values (dB)
    this_atten_values.push_back(0);
    this_atten_values.push_back(5);
    this_atten_values.push_back(10);
    this_atten_values.push_back(15);
    this_atten_values.push_back(20);
    this_atten_values.push_back(25);
    this_atten_values.push_back(30);

    return;    
}

//-- destructor ------------------------------------------------------------------------------------------------------------

sh_sm_radio::~sh_sm_radio( void )
{
    // clean up
    this->disconnect();

    return;
}

//-- public methods --------------------------------------------------------------------------------------------------------

// connect to USB radio
bool sh_sm_radio::connect_usb( void )
{
    int32_t      dev_ids[5]   = {};
    SmDeviceType dev_types[5] = {};
    int32_t      dev_cnt      = 0;
    bool         found        = false;
    SmStatus     status       = smNoError;

    // check if there are any USB radios
    status = smGetDeviceList2(dev_ids, dev_types, &dev_cnt);
    if ( status != smNoError ) { goto FAILED; }

    for ( int32_t i = 0; i < dev_cnt; i++ )
    {
        SmDeviceType dev_type = dev_types[i];

        if ( (dev_type == smDeviceTypeSM200A) || (dev_type == smDeviceTypeSM200B) || (dev_type == smDeviceTypeSM435B) )
        {
            this_device_id   = dev_ids[i];
            this_device_type = dev_types[i];
            this_device_name = this_device_names.at(this_device_type);

            if ( (dev_type == smDeviceTypeSM200A) || (dev_type == smDeviceTypeSM200B) )
            {
                this_center_freq_min_hz = (uint64_t)SM200_MIN_FREQ;
                this_center_freq_max_hz = (uint64_t)SM200_MAX_FREQ;
            }
            else // ( dev_type == smDeviceTypeSM435B )
            {
                this_center_freq_min_hz = (uint64_t)SM435_MIN_FREQ;
                this_center_freq_max_hz = (uint64_t)SM435_MAX_FREQ;
            }

            found = true;
            break;
        }
    }

    if ( ! found ) { goto FAILED; }

    // connect to the radio
    status = smOpenDevice(&this_device_handle);
    if ( status != smNoError ) { goto FAILED; }

    this_connected = true;

    return ( true );

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// connect to network radio
bool sh_sm_radio::connect_net( std::string ipaddr = SM_DEFAULT_ADDR, uint16_t port = SM_DEFAULT_PORT )
{
    return ( true );
}

//--------------------------------------------------------------------------------------------------------------------------

// connfigure radio
bool sh_sm_radio::configure( uint64_t sample_rate_hz, uint64_t center_freq_hz, int32_t atten_db = SM_AUTO_ATTEN,
                             double ref_level = -20.0 )
{
    double   sr_hz  = 0.0;
    double   bw_hz  = 0.0;
    bool     found  = false;
    SmStatus status = smNoError;

    // make sure the sample rate is supported
    for ( uint32_t i = 0; i < this_sample_rates.size(); i++ )
    {
        if ( this_sample_rates.at(i) == sample_rate_hz ) { found = true; break; }
    }

    if ( ! found ) { goto FAILED; }

    // make sure the attenuation value is supported
    found = false;

    if ( atten_db == SM_AUTO_ATTEN )
    {
        this_atten_value = SM_AUTO_ATTEN;
        found            = true;
    }
    else
    {
        for ( uint32_t i = 0; i < this_atten_values.size(); i++ )
        {
            if ( this_atten_values.at(i) == atten_db )
            {
                this_atten_value = i;
                found            = true;
                break;
            }
        }
    }

    if ( ! found ) { goto FAILED; }

    // set radio parameters
    smSetAttenuator(this_device_handle, this_atten_value);
    smSetRefLevel(this_device_handle, ref_level);
    smSetIQCenterFreq(this_device_handle, (double)center_freq_hz);
    smSetIQBaseSampleRate(this_device_handle, smIQStreamSampleRateNative);
    smSetIQSampleRate(this_device_handle, (uint32_t)(this_sample_rate_max_hz / sample_rate_hz));
    smSetIQDataType(this_device_handle, smDataType32fc);

    // configure the radio
    status = smConfigure(this_device_handle, smModeIQ);
    if ( status != smNoError) { goto FAILED; }

    // save settings
    smGetIQParameters(this_device_handle, &sr_hz, &bw_hz);

    this_sample_rate_hz  = sample_rate_hz;
    this_sample_rate_mhz = (double)((double)sample_rate_hz / 1000000.0);
    this_bandwidth_hz    = (uint64_t)bw_hz;
    this_bandwidth_mhz   = (double)(bw_hz / 1000000.0);
    this_center_freq_hz  = center_freq_hz;
    this_center_freq_mhz = (double)((double)center_freq_hz / 1000000.0);
    this_atten_db        = atten_db;
    this_ref_level       = ref_level;

    return ( true );

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// receive radio samples
bool sh_sm_radio::recv_samples( void *p_buffer, uint32_t buffer_len, int64_t *p_ns_since_epoch, bool *p_samples_dropped )
{
    uint32_t num_samples = (uint32_t)(buffer_len >> 3); // buffer_len / sizeof_sample (8 bytes)
    SmStatus status      = smNoError;

    // receive samples into buffer
    status = smGetIQ(this_device_handle, p_buffer, num_samples, nullptr, 0, p_ns_since_epoch,
                     /*purge*/smFalse, (int32_t *)p_samples_dropped, nullptr);
    if ( status != smNoError ) { goto FAILED; }

    return ( true );

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// disconnect from radio
void sh_sm_radio::disconnect( void )
{
    // clean up
    if ( this_connected )
    {
        smAbort(this_device_handle);
        smCloseDevice(this_device_handle);

        this_device_id     = -1;
        this_device_handle = -1;
        this_device_type   = -1;
        this_device_name   = std::string("");
        this_connected     = false;
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// getters
std::string sh_sm_radio::get_device_name( void )        { return ( this_device_name                      ); }
uint64_t    sh_sm_radio::get_sample_rate_hz( void )     { return ( this_sample_rate_hz                   ); }
double      sh_sm_radio::get_sample_rate_mhz( void )    { return ( this_sample_rate_mhz                  ); }
uint64_t    sh_sm_radio::get_sample_rate_max_hz( void ) { return ( this_sample_rate_max_hz               ); }
uint64_t    sh_sm_radio::get_bandwidth_hz( void )       { return ( this_bandwidth_hz                     ); }
double      sh_sm_radio::get_bandwidth_mhz( void )      { return ( this_bandwidth_mhz                    ); }
uint64_t    sh_sm_radio::get_center_freq_hz( void )     { return ( this_center_freq_hz                   ); }
double      sh_sm_radio::get_center_freq_mhz( void )    { return ( this_center_freq_mhz                  ); }
uint64_t    sh_sm_radio::get_center_freq_min_hz( void ) { return ( this_center_freq_min_hz               ); }
uint64_t    sh_sm_radio::get_center_freq_max_hz( void ) { return ( this_center_freq_max_hz               ); }
bool        sh_sm_radio::atten_is_auto( void )          { return ( this_atten_value == -1 ? true : false ); }
int32_t     sh_sm_radio::get_atten_db( void )           { return ( this_atten_db                         ); }
double      sh_sm_radio::get_ref_level( void )          { return ( this_ref_level                        ); }

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
