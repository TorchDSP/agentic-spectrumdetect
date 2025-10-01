#ifndef INCLUDE_TYE_BUFFER_POOL_H
#define INCLUDE_TYE_BUFFER_POOL_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_includes.h"
#include "tye_buffer.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class tye_buffer_pool
{
public: //==================================================================================================================

    // constructor(s) / destructor
    tye_buffer_pool( uint32_t num_buffers, uint32_t buffer_len, bool use_malloc = false );
   ~tye_buffer_pool( void );

    // public methods
    bool        create( void );
    void        destroy( void );
    tye_buffer* get( void );

private: //=================================================================================================================

    // private constants
    const std::string NAME = std::string("TYE_BUFFER_POOL");

    // private variables
    uint32_t this_num_buffers;
    uint32_t this_buffer_len;
    bool     this_use_malloc;
    bool     this_created;

    std::vector<tye_buffer *> this_buffers;

    // private methods
    void delete_buffers( void );
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_TYE_BUFFER_POOL_H
