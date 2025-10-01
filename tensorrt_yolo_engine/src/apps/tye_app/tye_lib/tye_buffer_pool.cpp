//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "tye_buffer_pool.h"

//-- constructor(s) --------------------------------------------------------------------------------------------------------

tye_buffer_pool::tye_buffer_pool( uint32_t num_buffers, uint32_t buffer_len, bool use_malloc )
{
    // initialize
    this_num_buffers = num_buffers;
    this_buffer_len  = buffer_len;
    this_use_malloc  = use_malloc;
    this_created     = false;

    this_buffers.clear();

    return;    
}

//-- destructor ------------------------------------------------------------------------------------------------------------

tye_buffer_pool::~tye_buffer_pool( void )
{
    // clean up
    this->destroy();

    return;
}

//-- public methods --------------------------------------------------------------------------------------------------------

// create buffer pool
bool tye_buffer_pool::create( void )
{
    bool ok = false;

    for ( uint32_t i = 0; i < this_num_buffers; i++ )
    {
        tye_buffer *p_buffer = new tye_buffer(this_buffer_len, this_use_malloc);
        if ( p_buffer == nullptr ) { goto FAILED; }

        ok = p_buffer->create();
        if ( ! ok ) { goto FAILED; }

        this_buffers.push_back(p_buffer);
    }

    this_created = true;

    return ( true );

FAILED:
    this->delete_buffers();

    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// destroy buffer pool
void tye_buffer_pool::destroy( void )
{
    // clean up
    if ( this_created )
    {
        this->delete_buffers();
        this_created = false;
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// get a buffer from the pool
tye_buffer* tye_buffer_pool::get( void )
{
    tye_buffer *p_buffer = nullptr;
    bool        locked   = false;

    for ( uint32_t i = 0; i < this_buffers.size(); i++ )
    {
        locked = this_buffers.at(i)->lock();
        if ( locked )
        {
            p_buffer = this_buffers.at(i);
            break;
        }
    }

    return ( p_buffer );
}

//-- private methods -------------------------------------------------------------------------------------------------------

// delete all buffers in the pool
void tye_buffer_pool::delete_buffers( void )
{
    for ( uint32_t i = 0; i < this_buffers.size(); i++ )
    {
        this_buffers.at(i)->destroy();
        delete this_buffers.at(i);
    }

    this_buffers.clear();

    return;
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
