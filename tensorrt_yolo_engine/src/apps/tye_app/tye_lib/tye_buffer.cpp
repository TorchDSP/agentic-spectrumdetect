//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "tye_buffer.h"

//-- constructor(s) --------------------------------------------------------------------------------------------------------

tye_buffer::tye_buffer( uint32_t len, bool use_malloc )
{
    // initialize
    this_use_malloc     = use_malloc;
    this_source_name    = std::string("");
    this_source_is_file = false;
    this_is_eof         = false;
    this_group_id       = 0;
    this_sequ_num       = 0;
    this_radio_retuned  = false;
    this_ns_since_epoch = 0;
    this_sample_rate_hz = 0;
    this_bandwidth_hz   = 0;
    this_center_freq_hz = 0;
    this_atten_db       = 0;
    this_ref_level      = 0.0;
    this_p_buffer       = nullptr;
    this_buffer_len     = len;
    this_locked         = false;
    this_created        = false;

    return;    
}

//-- destructor ------------------------------------------------------------------------------------------------------------

tye_buffer::~tye_buffer( void )
{
    // clean up
    this->destroy();

    return;
}

//-- public methods --------------------------------------------------------------------------------------------------------

// create buffer
bool tye_buffer::create( void )
{
    if ( this_use_malloc )
    {
        this_p_buffer = (uint8_t *)malloc(this_buffer_len);
        if ( this_p_buffer == nullptr ) { goto FAILED; }
    }
    else // allocate pinned memory
    {
        cudaError_t cuda_status = cudaMallocHost(&this_p_buffer, this_buffer_len);
        if ( this_p_buffer == nullptr ) { goto FAILED; }
    }

    this_created = true;

    return ( true );

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// destroy buffer
void tye_buffer::destroy( void )
{
    // clean up
    if ( this_created )
    {
        if ( this_use_malloc ) { free(this_p_buffer);         }
        else                   { cudaFreeHost(this_p_buffer); }

        this_p_buffer = nullptr;
        this_created  = false;
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// attempt to lock buffer
bool tye_buffer::lock( void )
{
    bool locked = false;

    this_mutex.lock();
    if ( ! this_locked ) { this_locked = locked = true; }
    this_mutex.unlock();

    return ( locked );
}

//--------------------------------------------------------------------------------------------------------------------------

// release buffer
void tye_buffer::release( void )
{
    this_mutex.lock();

    this_source_name    = std::string("");
    this_source_is_file = false;
    this_is_eof         = false;
    this_group_id       = 0;
    this_sequ_num       = 0;
    this_radio_retuned  = false;
    this_ns_since_epoch = 0;
    this_sample_rate_hz = 0;
    this_center_freq_hz = 0;
    this_center_freq_hz = 0;
    this_atten_db       = 0;
    this_ref_level      = 0.0;
    this_locked         = false;

    this_mutex.unlock();

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// set/get source name
void        tye_buffer::set_source_name( std::string source_name ) { this_source_name = source_name; }
std::string tye_buffer::get_source_name( void )                    { return ( this_source_name );    }

//--------------------------------------------------------------------------------------------------------------------------

// set/check if the source is a file
void tye_buffer::set_source_is_file( void ) { this_source_is_file = true;     }
bool tye_buffer::source_is_file( void )     { return ( this_source_is_file ); }

//--------------------------------------------------------------------------------------------------------------------------

// set/check if at the end of file
void tye_buffer::set_is_eof( void ) { this_is_eof = true;     }
bool tye_buffer::is_eof( void )     { return ( this_is_eof ); }

//--------------------------------------------------------------------------------------------------------------------------

// set/get group id
void     tye_buffer::set_group_id( uint64_t group_id ) { this_group_id = group_id; }
uint64_t tye_buffer::get_group_id( void )              { return ( this_group_id ); }

//--------------------------------------------------------------------------------------------------------------------------

// set/get sequence number
void     tye_buffer::set_sequ_num( uint64_t sequ_num ) { this_sequ_num = sequ_num; }
uint64_t tye_buffer::get_sequ_num( void )              { return ( this_sequ_num ); }

//--------------------------------------------------------------------------------------------------------------------------

// set/check radio retuned
void tye_buffer::set_radio_retuned( bool retuned ) { this_radio_retuned = retuned;  }
bool tye_buffer::radio_retuned( void )             { return ( this_radio_retuned ); }

//--------------------------------------------------------------------------------------------------------------------------

// set/get radio samples timestamp [nanoseconds since epoch]
void     tye_buffer::set_samples_ns_since_epoch( uint64_t ns ) { this_ns_since_epoch = ns;       }
uint64_t tye_buffer::get_samples_ns_since_epoch( void )        { return ( this_ns_since_epoch ); }

//--------------------------------------------------------------------------------------------------------------------------

// set/get radio sample rate
void     tye_buffer::set_sample_rate_hz( uint64_t rate_hz ) { this_sample_rate_hz = rate_hz;  }
uint64_t tye_buffer::get_sample_rate_hz( void )             { return ( this_sample_rate_hz ); }

//--------------------------------------------------------------------------------------------------------------------------

// set/get radio bandwidth
void     tye_buffer::set_bandwidth_hz( uint64_t freq_hz ) { this_bandwidth_hz = freq_hz;  }
uint64_t tye_buffer::get_bandwidth_hz( void )             { return ( this_bandwidth_hz ); }

//--------------------------------------------------------------------------------------------------------------------------

// set/get radio center frequency
void     tye_buffer::set_center_freq_hz( uint64_t freq_hz ) { this_center_freq_hz = freq_hz;  }
uint64_t tye_buffer::get_center_freq_hz( void )             { return ( this_center_freq_hz ); }

//--------------------------------------------------------------------------------------------------------------------------

// set/get radio attenuation
void    tye_buffer::set_atten_db( int32_t atten_db ) { this_atten_db = atten_db; }
int32_t tye_buffer::get_atten_db( void )             { return ( this_atten_db ); }

//--------------------------------------------------------------------------------------------------------------------------

// set/get radio reference level
void   tye_buffer::set_ref_level( double level ) { this_ref_level = level;    }
double tye_buffer::get_ref_level( void )         { return ( this_ref_level ); }

//--------------------------------------------------------------------------------------------------------------------------

// get buffer
uint8_t* tye_buffer::get( void ) { return ( this_p_buffer   ); }
uint32_t tye_buffer::len( void ) { return ( this_buffer_len ); }

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
