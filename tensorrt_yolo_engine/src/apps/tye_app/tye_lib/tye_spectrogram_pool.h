#ifndef INCLUDE_TYE_SPECTROGRAM_POOL_H
#define INCLUDE_TYE_SPECTROGRAM_POOL_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_includes.h"
#include "tye_spectrogram.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class tye_spectrogram_pool
{
public: //==================================================================================================================

    // constructor(s) / destructor
    tye_spectrogram_pool( uint32_t gpu, uint32_t fft_size, uint32_t num_spectrograms );
   ~tye_spectrogram_pool( void );

    // public methods
    bool             create( void );
    void             destroy( void );
    tye_spectrogram* get( void );

private: //=================================================================================================================

    // private constants
    const std::string NAME = std::string("TYE_SPECTROGRAM_POOL");

    // private variables
    uint32_t this_gpu;
    uint32_t this_fft_size;
    uint32_t this_num_spectrograms;
    bool     this_created;

    std::vector<tye_spectrogram *> this_spectrograms;

    // private methods
    void delete_spectrograms( void );
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_TYE_SPECTROGRAM_POOL_H
