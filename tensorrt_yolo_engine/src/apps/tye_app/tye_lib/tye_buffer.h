#ifndef INCLUDE_TYE_BUFFER_H
#define INCLUDE_TYE_BUFFER_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_includes.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class tye_buffer
{
public: //==================================================================================================================

    // constructor(s) / destructor
    tye_buffer( uint32_t len, bool use_malloc );
   ~tye_buffer( void );

    // public methods
    bool        create( void );
    void        destroy( void );
    bool        lock( void );
    void        release( void );
    void        set_source_name( std::string source_name );
    std::string get_source_name( void );
    void        set_source_is_file( void );
    bool        source_is_file( void );
    void        set_is_eof( void );
    bool        is_eof( void );
    void        set_group_id( uint64_t group_id );
    uint64_t    get_group_id( void );
    void        set_sequ_num( uint64_t sequ_num );
    uint64_t    get_sequ_num( void );
    void        set_radio_retuned( bool retuned );
    bool        radio_retuned( void );
    void        set_samples_ns_since_epoch( uint64_t ns );
    uint64_t    get_samples_ns_since_epoch( void );
    void        set_sample_rate_hz( uint64_t rate_hz );
    uint64_t    get_sample_rate_hz( void );
    void        set_bandwidth_hz( uint64_t freq_hz );
    uint64_t    get_bandwidth_hz( void );
    void        set_center_freq_hz( uint64_t freq_hz );
    uint64_t    get_center_freq_hz( void );
    void        set_atten_db( int32_t atten_db );
    int32_t     get_atten_db( void );
    void        set_ref_level( double level );
    double      get_ref_level( void );
    uint8_t*    get( void );
    uint32_t    len( void );

private: //=================================================================================================================

    // private constants
    const std::string NAME = std::string("TYE_BUFFER");

    // private variables
    std::mutex  this_mutex;
    bool        this_use_malloc;
    std::string this_source_name;
    bool        this_source_is_file;
    bool        this_is_eof;
    uint64_t    this_group_id;
    uint64_t    this_sequ_num;
    bool        this_radio_retuned;
    uint64_t    this_ns_since_epoch;
    uint64_t    this_sample_rate_hz;
    uint64_t    this_bandwidth_hz;
    uint64_t    this_center_freq_hz;
    int32_t     this_atten_db;
    double      this_ref_level;
    uint8_t    *this_p_buffer;
    uint32_t    this_buffer_len;
    bool        this_locked;
    bool        this_created;
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_TYE_BUFFER_H
