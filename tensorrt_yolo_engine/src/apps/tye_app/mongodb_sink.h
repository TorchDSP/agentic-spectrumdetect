#ifndef INCLUDE_MONGODB_SINK_H
#define INCLUDE_MONGODB_SINK_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_app_includes.h"
#include "tye_types.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class mongodb_sink
{
public: //==================================================================================================================

    // constructor(s) / destructor
    mongodb_sink( std::string creds, std::string name_ipaddr, uint16_t port );
   ~mongodb_sink( void );

    // public types
    typedef struct
    {
        uint64_t start_time_ns;
        double   duration_ns;
        uint64_t center_freq_hz;
        uint64_t bandwidth_hz;
        float    rssi;

    } detection;

    // public methods
    bool        connect( void );
    std::string get_connect_uri( void );
    std::string get_name( void );
    std::string get_collection_name( void );
    bool        insert( tye_types::detections_aux &aux, std::vector<mongodb_sink::detection> &detections );
    bool        update( uint64_t sequ_num, float spectrogram_diff );
    void        disconnect( void );

private: //=================================================================================================================

    // private constants
    const std::string NAME      = std::string("MONGODB_SINK");
    const uint32_t    OP_INSERT = 1;
    const uint32_t    OP_UPDATE = 2;

    // private types
    typedef struct
    {
        tye_types::detections_aux            detections_aux;
        std::vector<mongodb_sink::detection> detections;

    } op_insert;

    typedef struct
    {
        uint64_t sequ_num;
        float    spectrogram_diff;

    } op_update;

    typedef struct
    {
        uint32_t                type;
        mongodb_sink::op_insert insert;
        mongodb_sink::op_update update;

    } db_op;

    // private variables
    std::thread         *this_p_db_op_thread;
    std::string          this_db_creds;
    std::string          this_db_name_ipaddr;
    uint16_t             this_db_port;
    mongocxx::instance   this_db;
    mongocxx::uri        this_db_uri;
    mongocxx::client     this_db_client;
    std::string          this_db_name;
    mongocxx::database   this_db_connection;
    std::string          this_db_collection_name;
    mongocxx::collection this_db_collection;
    bool                 this_db_connected;
    bool                 this_db_disconnect;

    std::mutex                       this_db_op_mutex;
    std::vector<mongodb_sink::db_op> this_db_op_queue;

    // private methods
    void handle_db_op_insert( mongodb_sink::op_insert *p_insert );
    void handle_db_op_update( mongodb_sink::op_update *p_update );
    void db_op_thread( void );
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_MONGODB_SINK_H
