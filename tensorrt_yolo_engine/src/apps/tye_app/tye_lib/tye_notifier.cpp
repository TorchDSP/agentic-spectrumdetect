//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "tye_notifier.h"

//-- constructor(s) --------------------------------------------------------------------------------------------------------

tye_notifier::tye_notifier( void (*p_notify_cb)( tye_types::detections_tuple dt, void *p_data ), void *p_notify_cb_data )
{
    // initialize
    this_p_notify_cb      = p_notify_cb;
    this_p_notify_cb_data = p_notify_cb_data;
    this_p_mgr_thread     = nullptr;
    this_running          = false;
    this_ready            = false;
    this_exit             = false;
 
    return;    
}

//-- destructor ------------------------------------------------------------------------------------------------------------

tye_notifier::~tye_notifier( void ) { return; }

//-- public methods --------------------------------------------------------------------------------------------------------

// start the notifier
bool tye_notifier::start( void )
{
    this_p_mgr_thread = new std::thread(&tye_notifier::mgr_thread, this);
    if ( this_p_mgr_thread == nullptr ) { goto FAILED; }

    return ( true );

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// shutdown the notifier
void tye_notifier::shutdown( void )
{
    if ( this_running )
    {
        this_exit = true;
        this_p_mgr_thread->join();
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// check if the notifier is running/ready
bool tye_notifier::is_running( void ) { return ( this_running ); }
bool tye_notifier::is_ready( void )   { return ( this_ready   ); }

//--------------------------------------------------------------------------------------------------------------------------

// enqueue detections to the notifier
void tye_notifier::enqueue( tye_types::detections_tuple dt )
{
    this_mutex.lock();
    this_detections.push_back(dt);
    this_mutex.unlock();

    return;
}

//-- private methods -------------------------------------------------------------------------------------------------------

// notifier thread instance
void tye_notifier::mgr_thread( void )
{
    tye_types::detections_tuple detections      = tye_types::detections_tuple();
    bool                        have_detections = false;

    this_running = true;
    this_ready   = true;

    while ( true )
    {
        // exit requested ??
        if ( this_exit ) { break; }

        // attempt to dequeue detections
        have_detections = false;
        this_mutex.lock();

        if ( this_detections.size() > 0 )
        {
            detections = this_detections.at(0);
            this_detections.erase(this_detections.begin());

            have_detections = true;
        }
        this_mutex.unlock();

        // if we have detections...notify
        if ( have_detections ) { this_p_notify_cb(detections, this_p_notify_cb_data); }
        std::this_thread::yield();
    }

    this_running = false;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
