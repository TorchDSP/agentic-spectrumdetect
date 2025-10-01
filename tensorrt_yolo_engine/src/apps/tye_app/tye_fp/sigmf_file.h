#ifndef INCLUDE_SIGMF_FILE_H
#define INCLUDE_SIGMF_FILE_H

//--------------------------------------------------------------------------------------------------------------------------

#include "file_base.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class sigmf_file: public file_base
{
public: //==================================================================================================================

    // constructor(s) / destructor
    sigmf_file( std::string file_path );
   ~sigmf_file( void );

    // public types
    typedef struct
    {
        uint64_t sample_start;
        uint64_t sample_cnt;
        uint64_t freq_lower;
        uint64_t freq_upper;

    } signal;

    // public methods
    bool     load( void );
    uint64_t get_sample_rate( void );
    uint64_t get_center_freq( void );
    void     get_signals( std::vector<sigmf_file::signal> &signals );
    uint8_t* get_samples_at_signal( sigmf_file::signal signal, uint32_t fft_size, uint32_t *p_samples_len );
    bool     get_samples( uint8_t *p_buffer, uint32_t buffer_len );

private: //=================================================================================================================

    // private variables
    rapidjson::Document this_meta_json;
    std::ifstream       this_meta_file;
    std::ifstream       this_data_file;
    uint64_t            this_data_file_len;
    uint64_t            this_sample_rate;
    uint64_t            this_center_freq;

    std::vector<sigmf_file::signal> this_signals;
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_SIGMF_FILE_H
