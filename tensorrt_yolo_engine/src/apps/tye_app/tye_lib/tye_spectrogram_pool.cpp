//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "tye_spectrogram_pool.h"

//-- constructor(s) --------------------------------------------------------------------------------------------------------

tye_spectrogram_pool::tye_spectrogram_pool( uint32_t gpu, uint32_t fft_size, uint32_t num_spectrograms )
{
    // initialize
    this_gpu              = gpu;
    this_fft_size         = fft_size;
    this_num_spectrograms = num_spectrograms;
    this_created          = false;

    this_spectrograms.clear();

    return;    
}

//-- destructor ------------------------------------------------------------------------------------------------------------

tye_spectrogram_pool::~tye_spectrogram_pool( void )
{
    // clean up
    this->destroy();

    return;
}

//-- public methods --------------------------------------------------------------------------------------------------------

// create the spectrogram pool
bool tye_spectrogram_pool::create( void )
{
    bool ok = false;

    cudaSetDevice(this_gpu);

    for ( uint32_t i = 0; i < this_num_spectrograms; i++ )
    {
        tye_spectrogram *p_spectrogram = new tye_spectrogram(this_fft_size);
        if ( p_spectrogram == nullptr ) { goto FAILED; }

        ok = p_spectrogram->create();
        if ( ! ok ) { goto FAILED; }

        this_spectrograms.push_back(p_spectrogram);
    }

    this_created = true;

    return ( true );

FAILED:
    this->delete_spectrograms();

    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// destroy the spectrogram pool
void tye_spectrogram_pool::destroy( void )
{
    // clean up
    if ( this_created )
    {
        this->delete_spectrograms();
        this_created = false;
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// get a spectrogram from the pool
tye_spectrogram* tye_spectrogram_pool::get( void )
{
    tye_spectrogram *p_spectrogram = nullptr;
    bool             locked        = false;

    for ( uint32_t i = 0; i < this_spectrograms.size(); i++ )
    {
        locked = this_spectrograms.at(i)->lock();
        if ( locked )
        {
            p_spectrogram = this_spectrograms.at(i);
            break;
        }
    }

    return ( p_spectrogram );
}

//-- private methods -------------------------------------------------------------------------------------------------------

// delete all spectrograms in the pool
void tye_spectrogram_pool::delete_spectrograms( void )
{
    for ( uint32_t i = 0; i < this_spectrograms.size(); i++ )
    {
        this_spectrograms.at(i)->destroy();
        delete this_spectrograms.at(i);
    }

    this_spectrograms.clear();

    return;
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
