#ifndef INCLUDE_FILE_BASE_H
#define INCLUDE_FILE_BASE_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_app_includes.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class file_base
{
public: //==================================================================================================================

    // constructor(s) / destructor
    file_base( std::string file_path );
    virtual ~file_base( void );

    // public [virtual] methods
    virtual bool     load( void )                                          = 0;
    virtual uint64_t get_sample_rate( void )                               = 0;
    virtual uint64_t get_center_freq( void )                               = 0;
    virtual bool     get_samples( uint8_t *p_buffer, uint32_t buffer_len ) = 0;

    // public methods
    std::string get_file_path( void );

protected: //===============================================================================================================

    // protected variables
    std::string this_file_path;
    bool        this_is_loaded;

    // protected methods
    bool path_exists( const std::string &path );
    bool path_is_file( const std::string &path );
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_FILE_BASE_H
