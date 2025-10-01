#ifndef INCLUDE_FP_FRONTEND_H
#define INCLUDE_FP_FRONTEND_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_app_includes.h"
#include "tye_buffer.h"
#include "tye_buffer_pool.h"
#include "file_base.h"
#include "config.h"
#include "engine.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class fp_frontend
{
public: //==================================================================================================================

    // constructor(s) / destructor
    fp_frontend( tye_buffer_pool *p_buffer_pool, config *p_config, engine *p_engine );
   ~fp_frontend( void );

    // public methods
    bool start( void );
    void shutdown( void );
    bool is_running( void );

private: //=================================================================================================================

    // private constants
    const std::string NAME = std::string("FP_FRONTEND");

    // private variables
    std::thread     *this_p_mgr_thread;
    tye_buffer_pool *this_p_buffer_pool;
    config          *this_p_config;
    engine          *this_p_engine;
    std::string      this_path;
    bool             this_process_file;
    bool             this_process_dir;
    uint64_t         this_group_id;
    uint64_t         this_sequ_num;
    bool             this_running;
    bool             this_exit;

    // private methods
    bool path_exists( std::string &path );
    bool path_is_file( std::string &path );
    bool path_is_dir( std::string &path );
    void process_loaded_file( file_base *p_file );
    bool process_file( void );
    bool process_dir( void );
    void mgr_thread( void );
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_FP_FRONTEND_H
