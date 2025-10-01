#ifndef INCLUDE_TYE_SPECTROGRAM_H
#define INCLUDE_TYE_SPECTROGRAM_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_includes.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class tye_spectrogram
{
public: //==================================================================================================================

    // constructor(s) / destructor
    tye_spectrogram( uint32_t fft_size );
   ~tye_spectrogram( void );

    // public methods
    bool     create( void );
    void     destroy( void );
    bool     lock( void );
    void     release( void );
    void     set_sequ_num( uint64_t sequ_num );
    uint64_t get_sequ_num( void );
    float*   buffer( void );
    uint32_t buffer_len( void );

private: //=================================================================================================================

    // private constants
    const std::string NAME = std::string("TYE_SPECTROGRAM");

    // private variables
    std::mutex this_mutex;
    uint64_t   this_sequ_num;
    float     *this_p_buffer;
    uint32_t   this_buffer_len;
    bool       this_locked;
    bool       this_created;
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_TYE_SPECTROGRAM_H
