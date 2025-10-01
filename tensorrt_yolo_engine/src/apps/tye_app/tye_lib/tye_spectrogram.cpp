//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "tye_spectrogram.h"

//-- constructor(s) --------------------------------------------------------------------------------------------------------

tye_spectrogram::tye_spectrogram( uint32_t fft_size )
{
    // initialize
    this_sequ_num   = 0;
    this_p_buffer   = nullptr;
    this_buffer_len = (uint32_t)(fft_size * fft_size * sizeof(float));
    this_locked     = false;
    this_created    = false;

    return;    
}

//-- destructor ------------------------------------------------------------------------------------------------------------

tye_spectrogram::~tye_spectrogram( void )
{
    // clean up
    this->destroy();

    return;
}

//-- public methods --------------------------------------------------------------------------------------------------------

// create the spectrogram buffer
bool tye_spectrogram::create( void )
{
    this_p_buffer = (float *)malloc(this_buffer_len);
    if ( this_p_buffer == nullptr ) { goto FAILED; }

    this_created = true;

    return ( true );

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// destroy the spectrogram buffer
void tye_spectrogram::destroy( void )
{
    // clean up
    if ( this_created )
    {
        cudaFree(this_p_buffer);

        this_p_buffer = nullptr;
        this_created  = false;
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// attempt to lock the spectrogram buffer
bool tye_spectrogram::lock( void )
{
    bool locked = false;

    this_mutex.lock();
    if ( ! this_locked ) { this_locked = locked = true; }
    this_mutex.unlock();

    return ( locked );
}

//--------------------------------------------------------------------------------------------------------------------------

// release the spectrogram buffer
void tye_spectrogram::release( void )
{
    this_mutex.lock();

    this_sequ_num = 0;
    this_locked   = false;

    this_mutex.unlock();

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// set/get sequence number
void     tye_spectrogram::set_sequ_num( uint64_t sequ_num ) { this_sequ_num = sequ_num; }
uint64_t tye_spectrogram::get_sequ_num( void )              { return ( this_sequ_num ); }

//--------------------------------------------------------------------------------------------------------------------------

// get spectrogram buffer
float*   tye_spectrogram::buffer( void )     { return ( this_p_buffer   ); }
uint32_t tye_spectrogram::buffer_len( void ) { return ( this_buffer_len ); }

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
