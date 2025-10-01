//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "mongodb_sink.h"

//-- build conditionals ----------------------------------------------------------------------------------------------------

//#define MONGODB_SINK_CREATE_GROUP_ID_INDEX

//-- constructor(s) --------------------------------------------------------------------------------------------------------

mongodb_sink::mongodb_sink( std::string creds, std::string name_ipaddr, uint16_t port )
{
    // initialize
    this_p_db_op_thread     = nullptr;
    this_db_creds           = creds;
    this_db_name_ipaddr     = name_ipaddr;
    this_db_port            = port;
    this_db_collection_name = std::string("");
    this_db_connected       = false;
    this_db_disconnect      = false;

#if defined( TYE_STREAM_PROCESSOR )
    this_db_name = std::string("tye_sp");
#else
    this_db_name = std::string("tye_fp");
#endif

    this_db_op_queue.clear();

    if ( creds.empty() )
    {
        this_db_uri = mongocxx::uri("mongodb://" + name_ipaddr + ":" + std::to_string(port));
    }
    else
    {
        this_db_uri = mongocxx::uri("mongodb://" + this_db_creds + "@" + name_ipaddr + ":" + std::to_string(port));
    }

    return;
}

//-- destructor ------------------------------------------------------------------------------------------------------------

mongodb_sink::~mongodb_sink( void )
{
    // clean up
    this->disconnect();

    return;
}

//-- public methods --------------------------------------------------------------------------------------------------------

// connect to the database/sink
bool mongodb_sink::connect( void )
{
    // if not already connected to the database
    if ( ! this_db_connected )
    {
        try
        {
            // build the collection name based on the current date and time
            std::time_t now   = std::time(nullptr);
            std::tm    *p_now = std::localtime(&now);

            char date_time_str[32] = {};
            std::strftime(date_time_str, sizeof(date_time_str), "%Y%m%d_%H%M%S", p_now);

            this_db_collection_name = std::string("detections_") + std::string(date_time_str);

            // connect to the database
            this_db_client     = mongocxx::client(this_db_uri);
            this_db_connection = this_db_client[this_db_name];
            this_db_collection = this_db_connection[this_db_collection_name];

            // ping the database and check the result
            auto ping_cmd    = bsoncxx::builder::basic::make_document(bsoncxx::builder::basic::kvp("ping", 1));
            auto ping_result = this_db_connection.run_command(ping_cmd.view());

            std::string ping_result_json = bsoncxx::to_json(ping_result);
            if ( ping_result_json.find(std::string("ok")) == std::string::npos ) { goto FAILED; }

#ifdef MONGODB_SINK_CREATE_GROUP_ID_INDEX

            // add an index to the collection, on the group_id field
            bsoncxx::builder::basic::document index_builder;
            mongocxx::options::index          index_options;
        
            index_builder.append(bsoncxx::builder::basic::kvp("group_id", 1));
            bsoncxx::document::value index_key = index_builder.extract();
            index_options.unique(true);

            this_db_collection.create_index(index_key.view(), index_options);

#endif // MONGODB_SINK_CREATE_GROUP_ID_INDEX

            // start the database/sink operation handler thread
            this_p_db_op_thread = new std::thread(&mongodb_sink::db_op_thread, this);
            if ( this_p_db_op_thread == nullptr ) { goto FAILED; }

            this_db_connected = true;
        }
        catch ( mongocxx::exception &e ) { goto FAILED; }
    }

    return ( true );

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// get the URI used to connect to the database/sink
std::string mongodb_sink::get_connect_uri( void )
{
    if ( this_db_connected ) { return ( this_db_uri.to_string() ); }
    else                     { return ( std::string("")         ); }
}

//--------------------------------------------------------------------------------------------------------------------------

// get the database/sink name
std::string mongodb_sink::get_name( void )
{
    if ( this_db_connected ) { return ( this_db_name    ); }
    else                     { return ( std::string("") ); }
}

//--------------------------------------------------------------------------------------------------------------------------

// get the collection name associated with the connection
std::string mongodb_sink::get_collection_name( void )
{
    if ( this_db_connected ) { return ( this_db_collection_name ); }
    else                     { return ( std::string("")         ); }
}

//--------------------------------------------------------------------------------------------------------------------------

// enqueue a database/sink insert operation for processing
bool mongodb_sink::insert( tye_types::detections_aux &aux, std::vector<mongodb_sink::detection> &detections )
{
    mongodb_sink::db_op op = {};

    op.type                  = mongodb_sink::OP_INSERT;
    op.insert.detections_aux = aux;
    op.insert.detections     = detections;
    op.update                = mongodb_sink::op_update();

    this_db_op_mutex.lock();
    this_db_op_queue.push_back(op);
    this_db_op_mutex.unlock();

    return ( true );
}

//--------------------------------------------------------------------------------------------------------------------------

// enqueue a database/sink update operation for processing
bool mongodb_sink::update( uint64_t sequ_num, float spectrogram_diff )
{
    mongodb_sink::db_op op = {};

    op.type                    = mongodb_sink::OP_UPDATE;
    op.update.sequ_num         = sequ_num;
    op.update.spectrogram_diff = spectrogram_diff;
    op.insert                  = mongodb_sink::op_insert();

    this_db_op_mutex.lock();
    this_db_op_queue.push_back(op);
    this_db_op_mutex.unlock();

    return ( true );
}

//--------------------------------------------------------------------------------------------------------------------------

// disconnect from the database/sink
void mongodb_sink::disconnect( void )
{
    if ( this_db_connected )
    {
        this_db_disconnect = true;
        this_p_db_op_thread->join();

        this_db_connected = false;
    }

    return;
}

//-- private methods -------------------------------------------------------------------------------------------------------

// handle a database/sink insert operation
void mongodb_sink::handle_db_op_insert( mongodb_sink::op_insert *p_insert )
{
    uint32_t num_detections = (uint32_t)p_insert->detections.size();

    // if connected to the database
    if ( this_db_connected )
    {
        // generate the detections JSON payload
        rapidjson::Document rj_payload = {};
        rj_payload.SetObject();

        rapidjson::Document::AllocatorType &rj_allocator = rj_payload.GetAllocator();
        rapidjson::Value                    rj_detections(rapidjson::kArrayType);

        rj_payload.AddMember("group_id",             p_insert->detections_aux.group_id,             rj_allocator);
        rj_payload.AddMember("sequ_num",             p_insert->detections_aux.sequ_num,             rj_allocator);
        rj_payload.AddMember("ns_since_epoch",       p_insert->detections_aux.radio.ns_since_epoch, rj_allocator);
        rj_payload.AddMember("radio_sample_rate_hz", p_insert->detections_aux.radio.sample_rate_hz, rj_allocator);
        rj_payload.AddMember("radio_bandwidth_hz",   p_insert->detections_aux.radio.bandwidth_hz,   rj_allocator);
        rj_payload.AddMember("radio_center_freq_hz", p_insert->detections_aux.radio.center_freq_hz, rj_allocator);
        rj_payload.AddMember("radio_atten_db",       p_insert->detections_aux.radio.atten_db,       rj_allocator);
        rj_payload.AddMember("radio_ref_level",      p_insert->detections_aux.radio.ref_level,      rj_allocator);
        rj_payload.AddMember("num_detections",       num_detections,                                rj_allocator);

        for ( uint32_t i = 0; i < num_detections; i++ )
        {
            rapidjson::Value rj_detection(rapidjson::kObjectType);

            rj_detection.AddMember("start_time_ns",  p_insert->detections.at(i).start_time_ns,  rj_allocator);
            rj_detection.AddMember("duration_ns",    p_insert->detections.at(i).duration_ns,    rj_allocator);
            rj_detection.AddMember("center_freq_hz", p_insert->detections.at(i).center_freq_hz, rj_allocator);
            rj_detection.AddMember("bandwidth_hz",   p_insert->detections.at(i).bandwidth_hz,   rj_allocator);
            rj_detection.AddMember("rssi",           p_insert->detections.at(i).rssi,           rj_allocator);

            rj_detections.PushBack(rj_detection, rj_allocator);
        }

        rj_payload.AddMember("detections", rj_detections, rj_allocator);

        rapidjson::StringBuffer rj_payload_buffer = {};
        rapidjson::Writer<rapidjson::StringBuffer> rj_payload_writer(rj_payload_buffer);

        rj_payload_writer.SetMaxDecimalPlaces(6);
        rj_payload.Accept(rj_payload_writer);

        // write the detections JSON payload to the database
        try
        {
            bsoncxx::document::value db_bson = bsoncxx::from_json(rj_payload_buffer.GetString());
            this_db_collection.insert_one(db_bson.view());
        }
        catch ( mongocxx::exception &e )
        {
            std::cout << ">> " << mongodb_sink::NAME << " [EXCEPTION] => " << e.what() << " [FAIL]"
                      << std::endl << std::flush;
        }
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// handle a database/sink update operation
void mongodb_sink::handle_db_op_update( mongodb_sink::op_update *p_update )
{
    try
    {
        // update group with the spectrogram difference [best effort]
        bsoncxx::builder::basic::document query = {};
        query.append(bsoncxx::builder::basic::kvp("group_id", (int64_t)p_update->sequ_num));

        bsoncxx::builder::basic::document update = {};
        update.append(bsoncxx::builder::basic::kvp("$set", bsoncxx::builder::basic::make_document(
            bsoncxx::builder::basic::kvp("spectrogram_diff", p_update->spectrogram_diff))));

        auto result = this_db_collection.update_one(query.view(), update.view());
    }
    catch ( mongocxx::exception &e )
    {
        std::cout << ">> " << mongodb_sink::NAME << " [EXCEPTION] => " << e.what() << " [FAIL]"
                  << std::endl << std::flush;
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// database/sink operation handler thread
void mongodb_sink::db_op_thread( void )
{
    uint32_t op_cnt = 0;

    // run until disconnect requested
    while ( true )
    {
        // disconnect requested ?? only bail after all database/sink operations have been processed
        if ( this_db_disconnect ) { break; }

        // attempt to dequeue a database/sink operation
        mongodb_sink::db_op op = {};
        bool                ok = false;

        this_db_op_mutex.lock();

        if ( this_db_op_queue.size() > 0 )
        {
            op = this_db_op_queue.at(0);
            ok = true;

            this_db_op_queue.erase(this_db_op_queue.begin());
        }
        this_db_op_mutex.unlock();

        // handle database/sink operation
        if ( ok )
        {
            if      ( op.type == mongodb_sink::OP_INSERT ) { this->handle_db_op_insert(&op.insert); }
            else if ( op.type == mongodb_sink::OP_UPDATE ) { this->handle_db_op_update(&op.update); }
        }

        std::this_thread::yield();
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
