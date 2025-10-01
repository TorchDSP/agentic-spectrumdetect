#ifndef INCLUDE_TYE_NOTIFIER_H
#define INCLUDE_TYE_NOTIFIER_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_includes.h"
#include "tye_types.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class tye_notifier
{
public: //==================================================================================================================

    // constructor(s) / destructor
    tye_notifier( void (*p_notify_cb)( tye_types::detections_tuple dt, void *p_data ), void *p_notify_cb_data );
   ~tye_notifier( void );

    // public methods
    bool start( void );
    void shutdown( void );
    bool is_running( void );
    bool is_ready( void );
    void enqueue( tye_types::detections_tuple dt );

private: //=================================================================================================================

    // private constants
    const std::string NAME = std::string("TYE_NOTIFIER");

    // private variables
    void (*this_p_notify_cb)( tye_types::detections_tuple dt, void *p_data );
    void  *this_p_notify_cb_data;

    std::thread *this_p_mgr_thread;
    std::mutex   this_mutex;
    bool         this_running;
    bool         this_ready;
    bool         this_exit;

    std::vector<tye_types::detections_tuple> this_detections;

    // private methods
    void mgr_thread( void );
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_TYE_NOTIFIER_H
