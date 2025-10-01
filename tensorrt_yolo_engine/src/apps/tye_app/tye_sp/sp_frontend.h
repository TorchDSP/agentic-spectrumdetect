#ifndef INCLUDE_SP_FRONTEND_H
#define INCLUDE_SP_FRONTEND_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_app_includes.h"
#include "tye_buffer.h"
#include "tye_buffer_pool.h"
#include "config.h"
#include "sh_sm_radio.h"
#include "engine.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class sp_frontend
{
public: //==================================================================================================================

    // constructor(s) / destructor
    sp_frontend( tye_buffer_pool *p_buffer_pool, config *p_config, engine *p_engine );
   ~sp_frontend( void );

    // public methods
    bool start( void );
    void shutdown( void );
    bool is_running( void );

private: //=================================================================================================================

    // private constants
    const std::string NAME = std::string("SP_FRONTEND");

    // private types
    typedef enum
    {
        CONNECT_RADIO,
        SETUP_RETUNE,
        CONFIGURE_RADIO,
        GET_RECV_BUFFER,
        RECV_SAMPLES,
        HANDLE_RETUNE,
        FAILED,
        IDLE

    } state;

    // private variables
    sh_sm_radio       *this_p_radio;
    std::thread       *this_p_mgr_thread;
    tye_buffer        *this_p_stream_buffer;
    tye_buffer_pool   *this_p_buffer_pool;
    config            *this_p_config;
    engine            *this_p_engine;
    sp_frontend::state this_state;
    int32_t            this_retune_socket;
    bool               this_retuned;
    uint64_t           this_group_id;
    uint64_t           this_sequ_num;
    bool               this_running;
    bool               this_connected;
    bool               this_exit;

    // private methods
    void clean_up( void );
    void send_retune_status( bool success, struct sockaddr_in *p_peer_sa_in, ssize_t peer_sa_in_len );
    void handle_state_connect_radio( void );
    void handle_state_setup_retune( void );
    void handle_state_configure_radio( void );
    void handle_state_get_recv_buffer( void );
    void handle_state_recv_samples( void );
    void handle_state_handle_retune( void );
    void handle_state_disconnect_radio( void );
    void handle_state_failed( void );
    void handle_state_idle( void );
    void mgr_thread( void );
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_SP_FRONTEND_H
