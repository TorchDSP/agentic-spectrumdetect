//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "engine.h"

//--------------------------------------------------------------------------------------------------------------------------

//#define ENGINE_PROCESS_STREAM_IMAGES_PER_SEC

//-- constructor(s) --------------------------------------------------------------------------------------------------------

engine::engine( config *p_config, mongodb_sink *p_mongodb_sink, bool test_debug )
{
    // initialize
    this_exit_cb             = nullptr;
    this_p_mgr_thread        = nullptr;
    this_p_dproc_thread      = nullptr;
    this_p_ad_thread         = nullptr;
    this_p_config            = p_config;
    this_p_mongodb_sink      = p_mongodb_sink;
    this_p_multiplot         = nullptr;
    this_p_fft_bins_proc     = nullptr;
    this_stream              = {};
    this_cnc_socket          = -1;
    this_gl_renderer_name    = std::string("");
    this_is_sw_gl_renderer   = false;
    this_test_debug          = test_debug;
    this_process_stream      = false;
    this_database_off        = this_p_config->database_off();
    this_engines_per_gpu     = this_p_config->engines_per_gpu();
    this_output_path         = this_p_config->output_path();
    this_engine_cluster_idx  = 0;
    this_last_buffer_millis  = 0;
    this_buffer_flow_started = false;
    this_running             = false;
    this_ad_ready            = false;
    this_dproc_ready         = false;
    this_engine_ready        = false;
    this_exit                = false;

    this_buffer_queue.clear();
    this_detection_groups_in.clear();
    this_detection_groups_out.clear();
    this_detections_batch.sequ_nums.clear();
    this_detections_batch.images.clear();
    this_detections_batch.spectrograms.clear();
    this_detections_batch.spectrogram_diffs.clear();
    this_engine_clusters.clear();

    return;    
}

//-- destructor ------------------------------------------------------------------------------------------------------------

engine::~engine( void )
{
    // clean up
    this->shutdown();

    return;
}

//-- public methods --------------------------------------------------------------------------------------------------------

// start engine
bool engine::start( void (*exit_cb)( void ) )
{
    std::vector<uint32_t> gpu               = std::vector<uint32_t>();
    tye_cluster          *p_engine_cluster  = nullptr;
    uint32_t              fft_bins_proc_gpu = this_p_config->gpus().at(0);
    time_t                ready_timeout     = 0;
    cudaError_t           cuda_status       = cudaSuccess;
    int32_t               status            = -1;
    bool                  ok                = false;

    // start the tensor-rt yolo engine clusters...one per gpu
    uint32_t num_gpus = (uint32_t)this_p_config->gpus().size();

    for ( uint32_t i = 0; i < num_gpus; i++ )
    {
        gpu.clear();
        gpu.push_back(this_p_config->gpus().at(i));

        p_engine_cluster = new tye_cluster(gpu, this_p_config->engine_path(), this_p_config->engines_per_gpu(),
                                           this_p_config->fft_size(), this_p_config->score_threshold(),
                                           this_p_config->nms_threshold(), this_p_config->power_offset(),
                                           this_p_config->pixel_min_val(), this_p_config->pixel_max_val(),
                                           this_p_config->false_detect_w(), this_p_config->false_detect_h(),
                                           notify_detections_cb, this, this_test_debug);
        if ( p_engine_cluster == nullptr ) { goto FAILED; }

        ok = p_engine_cluster->start();
        if ( ! ok ) { goto FAILED; }

        this_engine_clusters.push_back(p_engine_cluster);
    }

    // open command and control socket
    ok = this->open_cnc_socket();
    if ( ! ok ) { goto FAILED; }

    // create a CUDA stream for FFT bins processing
    cudaSetDevice(fft_bins_proc_gpu);

    cuda_status = cudaStreamCreate(&this_stream);
    if ( cuda_status != cudaSuccess ) { goto FAILED_CLOSE_CNC_SOCKET; }

    // create FFT bins processor
    this_p_fft_bins_proc = new cuda_fft_bins_proc(fft_bins_proc_gpu, (this_p_config->fft_size() * this_p_config->fft_size()));
    if ( this_p_fft_bins_proc == nullptr ) { goto FAILED_DESTROY_STREAM; }

    this_process_stream = this_p_config->process_stream();

    // start the engine advertisement thread
    this_p_ad_thread = new std::thread(&engine::ad_thread, this);
    if ( this_p_ad_thread == nullptr ) { goto FAILED_FREE_FFT_BINS_PROC; }

    // start the engine detections processor thread
    this_p_dproc_thread = new std::thread(&engine::dproc_thread, this);
    if ( this_p_dproc_thread == nullptr ) { goto FAILED_SHUTDOWN_AD_THREAD; }

    // start the engine manager thread
    this_p_mgr_thread = new std::thread(&engine::mgr_thread, this);
    if ( this_p_mgr_thread == nullptr ) { goto FAILED_SHUTDOWN_DPROC_THREAD; }

    // wait, with timeout, for the detections processor and engine to be ready
    ready_timeout = (time(nullptr) + (time_t)2000);

    while ( true )
    {
        if ( time(nullptr)     > ready_timeout     ) { goto FAILED_SHUTDOWN_MGR_THREAD; }
        if ( this_dproc_ready && this_engine_ready ) { break;                           }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    this_exit_cb = exit_cb;

    return ( true );

FAILED_SHUTDOWN_MGR_THREAD:
    this_exit = true;
    this_p_mgr_thread->join();

FAILED_SHUTDOWN_DPROC_THREAD:
    this_exit = true;
    this_p_dproc_thread->join();

FAILED_SHUTDOWN_AD_THREAD:
    this_exit = true;
    this_p_ad_thread->join();

FAILED_FREE_FFT_BINS_PROC:
    delete this_p_fft_bins_proc;
    this_p_fft_bins_proc = nullptr;

FAILED_DESTROY_STREAM:
    cudaStreamDestroy(this_stream);

FAILED_CLOSE_CNC_SOCKET:
    this->close_cnc_socket();

FAILED:
    if ( p_engine_cluster != nullptr ) { delete p_engine_cluster; }
    this->shutdown_engine_clusters();

    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// shutdown engine
void engine::shutdown( void )
{
    // clean up
    if ( this_running )
    {
        this->close_cnc_socket();

        this_exit = true;
        this_p_ad_thread->join();
        this_p_mgr_thread->join();
        this_p_dproc_thread->join();

        delete this_p_fft_bins_proc;

        cudaStreamSynchronize(this_stream);
        cudaStreamDestroy(this_stream);

        this->shutdown_engine_clusters();
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// check if the engine is running
bool engine::is_running( void ) { return ( this_running ); }

//--------------------------------------------------------------------------------------------------------------------------

// getters
std::string engine::get_gl_renderer_name( void ) { return ( this_gl_renderer_name ); }

//--------------------------------------------------------------------------------------------------------------------------

// process a buffer of samples
void engine::process( tye_buffer *p_buffer )
{
    // no harm in setting this every time
    this_buffer_flow_started = true;

    // if start of a new file
    if ( (p_buffer->get_sequ_num() == 0) && p_buffer->source_is_file() )
    {
        this_detections_in_mutex.lock();
        std::cout << ">> " << engine::NAME << " => PROCESSING FILE [" << p_buffer->get_source_name() << "]"
                  << std::endl << std::flush;
        this_detections_in_mutex.unlock();
    }

    // queue up, and dispatch buffers to tensor-rt yolo engine clusters in batches
    //
    // ex. if each tensor-rt yolo engine cluster was started with 4 instances of the tensor-rt yolo engine, then
    //     queue up 4 buffers and dispatch them to a tensor-rt yolo engine cluster in one shot, in an effort to
    //     maximize parallel processing
    this_buffer_mutex.lock();
    this_last_buffer_millis = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::
                                             now().time_since_epoch()).count();

    // if the radio was retuned then flush the current batch
    if ( p_buffer->radio_retuned() )
    {
        uint32_t num_buffers = (uint32_t)this_buffer_queue.size();
        for ( uint32_t i = 0; i < num_buffers; i++ ) { this_buffer_queue.at(i)->release(); }
        this_buffer_queue.clear();

        if ( this_p_multiplot != nullptr ) { this_p_multiplot->flush_history(); }
    }
    // else if processing a file and end-of-file reached
    else if ( p_buffer->source_is_file() && p_buffer->is_eof() )
    {
        // enqueue pending buffers to a tensor-rt yolo engine cluster for processing
        for ( uint32_t i = 0; i < this_buffer_queue.size(); i++ )
        {
            tye_buffer *p_buffer = this_buffer_queue.at(i);
            this_engine_clusters.at(this_engine_cluster_idx)->process(p_buffer);
        }
        this_buffer_queue.clear();

        // round robin across tensor-rt yolo engine clusters
        this_engine_cluster_idx++;
        if ( this_engine_cluster_idx == this_engine_clusters.size() ) { this_engine_cluster_idx = 0; }
    }
    // else enqueue buffer for processing
    else { this_buffer_queue.push_back(p_buffer); }

    if ( this_buffer_queue.size() == this_engines_per_gpu ) // have a complete batch ??
    {
        // enqueue batch of buffers to a tensor-rt yolo engine cluster for processing
        for ( uint32_t i = 0; i < this_engines_per_gpu; i++ )
        {
            tye_buffer *p_buffer = this_buffer_queue.at(i);
            this_engine_clusters.at(this_engine_cluster_idx)->process(p_buffer);
        }
        this_buffer_queue.clear();

        // round robin across tensor-rt yolo engine clusters
        this_engine_cluster_idx++;
        if ( this_engine_cluster_idx == this_engine_clusters.size() ) { this_engine_cluster_idx = 0; }
    }
    this_buffer_mutex.unlock();

    return;
}

//-- private [static] methods ----------------------------------------------------------------------------------------------

// callback to handle notification of detections...provided to tensor-rt yolo egnine clusters, and called
// each time a buffer has been processed, resulting in detections, etc
void engine::notify_detections_cb( tye_types::detections_tuple dt, void *p_data )
{
    engine                           *p_engine         = (engine *)p_data;
    std::vector<tye_types::detection> detections       = std::get<0>(dt);
    tye_types::detections_aux         detections_aux   = std::get<1>(dt);
    cv::Mat                           detections_image = std::get<2>(dt);
    tye_spectrogram                  *p_spectrogram    = std::get<3>(dt);
    uint64_t                          group_id         = detections_aux.group_id;
    std::string                       source_name      = detections_aux.source_name;
    std::string                       image_path       = std::string("");
    bool                              found            = false;

    // writing output ??
    if ( ! p_engine->this_output_path.empty() )
    {
        std::filesystem::path fs_path(source_name);
        image_path = p_engine->this_output_path + "/" + fs_path.filename().string();
    }

    p_engine->this_detections_in_mutex.lock();

    // check if these detections belong to an existing group
    uint32_t num_detection_groups = (uint32_t)p_engine->this_detection_groups_in.size();

    for ( uint32_t i = 0; i < num_detection_groups; i++ )
    {
        engine::detection_group *p_group = p_engine->this_detection_groups_in.at(i);

        // these detections belong to this group
        if ( p_group->id == group_id )
        {
            // update the last time detections were seen on this group
            p_group->last_millis = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::
                                                  system_clock::now().time_since_epoch()).count();
            // add detections to group
            p_group->detection_data.push_back(std::make_tuple(detections, detections_aux, detections_image, p_spectrogram));

            found = true;
            break;
        }
    }

    // these detections do not belong to an existing group
    if ( ! found )
    {
        // create a new group and add detections
        engine::detection_group *p_group = new engine::detection_group();
        if ( p_group != nullptr )
        {
            p_group->id           = group_id;
            p_group->source_name  = source_name;
            p_group->start_millis = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::
                                                   system_clock::now().time_since_epoch()).count();
            p_group->last_millis  = p_group->start_millis;
            p_group->image_path   = image_path;

            p_group->detection_data.push_back(std::make_tuple(detections, detections_aux, detections_image, p_spectrogram));
            p_engine->this_detection_groups_in.push_back(p_group);
        }
    }

    p_engine->this_detections_in_mutex.unlock();

    return;
}

//-- private methods -------------------------------------------------------------------------------------------------------

// open command and control socket
bool engine::open_cnc_socket( void )
{
    struct sockaddr_in bind_addr    = {};
    int32_t            reuseaddr_on =  1;
    int32_t            status       = -1;

    // create a UDP socket and bind it to the requested command and control port
    this_cnc_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if ( this_cnc_socket == -1 ) { goto FAILED; }

    fcntl(this_cnc_socket, F_SETFL, fcntl(this_cnc_socket, F_GETFL, 0) | O_NONBLOCK); // non-blocking

    status = setsockopt(this_cnc_socket, SOL_SOCKET, SO_REUSEADDR, &reuseaddr_on, sizeof(reuseaddr_on));
    if ( status == -1 ) { goto FAILED_CLOSE_SOCKET; }

    std::memset(&bind_addr, 0, sizeof(bind_addr));

    bind_addr.sin_family      = AF_INET;
    bind_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    bind_addr.sin_port        = htons(this_p_config->cnc_port());

    status = bind(this_cnc_socket, (struct sockaddr *)&bind_addr, sizeof(bind_addr));
    if ( status == -1 ) { goto FAILED_CLOSE_SOCKET; }

    return ( true );

FAILED_CLOSE_SOCKET:
    close(this_cnc_socket);
    this_cnc_socket = -1;

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// send command and control status
void engine::send_cnc_status( bool success, std::string msg_type, struct sockaddr_in *p_peer_sa_in, ssize_t peer_sa_in_len )
{
    rapidjson::Document rj_msg = {};
    rj_msg.SetObject();

    rapidjson::Document::AllocatorType &rj_allocator = rj_msg.GetAllocator();
    rapidjson::Value                    rj_msg_type(msg_type.c_str(), rj_allocator);

    rj_msg.AddMember("msg_type", rj_msg_type, rj_allocator);
    if ( success ) { rj_msg.AddMember("status", "success", rj_allocator); }
    else           { rj_msg.AddMember("status", "failure", rj_allocator); }

    rapidjson::StringBuffer rj_msg_buffer = {};
    rapidjson::Writer<rapidjson::StringBuffer> rj_msg_writer(rj_msg_buffer);

    rj_msg.Accept(rj_msg_writer);

    sendto(this_cnc_socket, rj_msg_buffer.GetString(), strlen(rj_msg_buffer.GetString()), 0,
           (struct sockaddr *)p_peer_sa_in, peer_sa_in_len);

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// handle command and control - update parameters
void engine::handle_cnc_update_params( rapidjson::Document &rj_cnc, struct sockaddr_in *p_peer_sa_in,
                                       ssize_t peer_sa_in_len )
{
    std::string msg_type = std::string("update_params_status");

    // verify all required keys/values are present
    if ( ! rj_cnc.HasMember("power_offset") )
    {
        std::cout << "[KEY \"power_offset\" NOT FOUND] " << std::flush;
        goto FAILED;
    }

    if ( ! rj_cnc.HasMember("pixel_min_val") )
    {
        std::cout << "[KEY \"pixel_min_val\" NOT FOUND] " << std::flush;
        goto FAILED;
    }

    if ( ! rj_cnc.HasMember("pixel_max_val") )
    {
        std::cout << "[KEY \"pixel_max_val\" NOT FOUND] " << std::flush;
        goto FAILED;
    }

    if ( ! rj_cnc.HasMember("false_detect_w") )
    {
         std::cout << "[KEY \"false_detect_w\" NOT FOUND] " << std::flush;
         goto FAILED;
    }

    if ( ! rj_cnc.HasMember("false_detect_h") )
    {
        std::cout << "[KEY \"false_detect_h\" NOT FOUND] " << std::flush;
        goto FAILED;
    }

    {
        // extract values
        float    power_offset   = (float)rj_cnc["power_offset"].GetFloat();
        float    pixel_min_val  = (float)rj_cnc["pixel_min_val"].GetFloat();
        float    pixel_max_val  = (float)rj_cnc["pixel_max_val"].GetFloat();
        uint32_t false_detect_w = (uint32_t)rj_cnc["false_detect_w"].GetUint();
        uint32_t false_detect_h = (uint32_t)rj_cnc["false_detect_h"].GetUint();

        // push parameter updates to the engine clusters
        for ( uint32_t i = 0; i < this_engine_clusters.size(); i++ )
        {
            this_engine_clusters.at(i)->update_params(power_offset, pixel_min_val, pixel_max_val,
                                                      false_detect_w, false_detect_h);
        }

        // update configuration
        this_p_config->set_power_offset(power_offset);
        this_p_config->set_pixel_min_val(pixel_min_val);
        this_p_config->set_pixel_max_val(pixel_max_val);
        this_p_config->set_false_detect_w(false_detect_w);
        this_p_config->set_false_detect_h(false_detect_h);

        std::cout << "[OK] POWER OFFSET [" << power_offset << "] PIXEL MIN VALUE [" << pixel_min_val
                  << "] PIXEL MAX VALUE [" << pixel_max_val << "] FALSE DETECT WIDTH [" << false_detect_w
                  << "] FALSE DETECT HEIGHT [" << false_detect_h << "]" << std::endl << std::flush;

        // send success status
        this->send_cnc_status(true, msg_type, p_peer_sa_in, peer_sa_in_len);
    }

    return;

FAILED:
    std::cout << "[FAIL]" << std::endl << std::flush;

    // send failure status
    this->send_cnc_status(false, msg_type, p_peer_sa_in, peer_sa_in_len);

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// handle command and control
void engine::handle_cnc( void )
{
    char               cnc[1024]      = {};
    struct sockaddr_in peer_sa_in     = {};
    socklen_t          peer_sa_in_len = sizeof(peer_sa_in);
    ssize_t            bytes_recvd    = 0;

    std::memset(&peer_sa_in, 0, sizeof(peer_sa_in));

    // attempt to receve command and control
    bytes_recvd = recvfrom(this_cnc_socket, cnc, sizeof(cnc), 0, (struct sockaddr *)&peer_sa_in,
                           &peer_sa_in_len);
    if ( bytes_recvd > 0 )
    {
        std::cout << ">> " << engine::NAME << " => HANDLING CNC " << std::flush;

        rapidjson::Document rj_cnc;
        rj_cnc.Parse(cnc);

        // parse the message
        if ( rj_cnc.HasParseError() )
        {
            std::cout << "[PARSE ERROR] " << std::flush;
            goto FAILED;
        }

        if ( ! rj_cnc.IsObject() )
        {
            std::cout << "[MSG IS NOT AN OBJECT] " << std::flush;
            goto FAILED;
        }

        // handle based on message type
        if ( ! rj_cnc.HasMember("msg_type") )
        {
            std::cout << "[KEY \"msg_type\" NOT FOUND] " << std::flush;
            goto FAILED;
        }

        std::string msg_type = rj_cnc["msg_type"].GetString();
        if ( msg_type.compare("update_params") == 0 )
        {
            this->handle_cnc_update_params(rj_cnc, &peer_sa_in, peer_sa_in_len);
        }
        else
        {
            std::cout << "[MSG TYPE \"" << msg_type << "\" INVALID] " << std::flush;
            goto FAILED;
        }
    }

    return;

FAILED:
    std::cout << "[FAIL]" << std::endl << std::flush;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// close command and control socket
void engine::close_cnc_socket( void )
{
    if ( this_cnc_socket != -1 )
    {
        close(this_cnc_socket);
        this_cnc_socket = -1;
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// shutdown all engine clusters
void engine::shutdown_engine_clusters( void )
{
    uint32_t num_engine_clusters = (uint32_t)this_engine_clusters.size();

    for ( uint32_t i = 0; i < num_engine_clusters; i++ )
    {
        this_engine_clusters.at(i)->shutdown();
        delete this_engine_clusters.at(i);
    }

    this_engine_clusters.clear();

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// handle detections groups that are done [stream processing]
void engine::handle_detection_groups_done_stream( void )
{
    this_detections_in_mutex.lock();

    // for each detection group
    uint32_t num_detection_groups_in = this_detection_groups_in.size();

    for ( uint32_t i = 0; i < num_detection_groups_in; i++ )
    {
        // enqueue the detection group for processing
        this_detections_out_mutex.lock();
        this_detection_groups_out.push_back(this_detection_groups_in.at(i));
        this_detections_out_mutex.unlock();
    }
    this_detection_groups_in.clear();

    this_detections_in_mutex.unlock();

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// handle detections groups that are done [file processing]
void engine::handle_detection_groups_done_file( void )
{
    this_detections_in_mutex.lock();
    uint64_t now_millis = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::
                                         now().time_since_epoch()).count();
    // for each detection group
    uint32_t num_detection_groups_in = this_detection_groups_in.size();

    for ( uint32_t i = 0; i < num_detection_groups_in; i++ )
    {
        engine::detection_group *p_group = this_detection_groups_in.at(i);

        // if no new detections added to this detection group in a bit, then we'll call it done
        if ( (now_millis - p_group->last_millis) > 1000/*millis*/ )
        {
            uint64_t processing_millis = (p_group->last_millis - p_group->start_millis);

            std::cout << ">> " << engine::NAME << " => FILE PROCESSED [" << p_group->source_name << "] PROCESSING TIME ["
                      << processing_millis << " MILLIS]" << std::endl << std::flush;

            // enqueue the detection group for processing
            this_detections_out_mutex.lock();
            this_detection_groups_out.push_back(p_group);
            this_detections_out_mutex.unlock();

            this_detection_groups_in.erase(this_detection_groups_in.begin() + i);
            break;
        }
    }

    this_detections_in_mutex.unlock();

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// check if file processing is done
void engine::check_file_processing_done( void )
{
    bool processing_done = false;

    // if the buffer flow has started
    if ( this_buffer_flow_started )
    {
        // and we are processing files
        if ( ! this_process_stream )
        {
            // get the number of detection groups
            this_detections_in_mutex.lock();
            uint32_t num_detection_groups = this_detection_groups_in.size();
            this_detections_in_mutex.unlock();

            this_buffer_mutex.lock();
            uint64_t now_millis = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::
                                                 system_clock::now().time_since_epoch()).count();

            // if no new buffers have been enqueued in a bit, and all detections groups are done
            if ( ((now_millis - this_last_buffer_millis) > 1000/*millis*/) && (num_detection_groups == 0) )
            {
                // release pending buffers...should never hit this
                uint32_t num_buffers = (uint32_t)this_buffer_queue.size();

                for ( uint32_t i = 0; i < num_buffers; i++ )
                {
                    tye_buffer *p_buffer = this_buffer_queue.at(i);
                    p_buffer->release();
                }
                this_buffer_queue.clear();

                std::cout << ">> " << engine::NAME << " => PROCESSING DONE" << std::endl << std::flush;
                this_buffer_flow_started = false;

                // processing is done
                processing_done = true;
            }
            this_buffer_mutex.unlock();
        }
    }

    if ( processing_done ) { this_exit_cb(); }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// check if there are detection groups to process
bool engine::detection_groups_to_process( void )
{
    bool have_groups_to_process = false;

    this_detections_out_mutex.lock();
    if ( this_detection_groups_out.size() > 0 ) { have_groups_to_process = true; }
    this_detections_out_mutex.unlock();

    return ( have_groups_to_process );
}

//--------------------------------------------------------------------------------------------------------------------------

// generate a list of database detections
void engine::generate_db_detections( tye_types::detections_aux &detections_aux, std::vector<tye_types::detection> &detections,
                                     std::vector<mongodb_sink::detection> &db_detections )
{
    uint64_t group_id       = detections_aux.group_id;
    uint64_t ns_since_epoch = detections_aux.radio.ns_since_epoch;
    double   sample_rate_hz = (double)detections_aux.radio.sample_rate_hz;
    double   center_freq_hz = (double)detections_aux.radio.center_freq_hz;
    double   fft_size       = (double)this_p_config->fft_size();
    double   ns_per_pixel   = (double)(((1.0 / sample_rate_hz) * 1000000000.0) / fft_size);
    double   min_freq_hz    = (double)(center_freq_hz - (sample_rate_hz / 2.0));
    double   max_freq_hz    = (double)(center_freq_hz + (sample_rate_hz / 2.0));
    double   hz_per_pixel   = (double)((max_freq_hz - min_freq_hz) / fft_size);
    uint32_t num_detections = (uint32_t)detections.size();

    for ( uint32_t i = 0; i < num_detections; i++ )
    {
        tye_types::detection *p_detection = &detections.at(i);
        double                bbox_x      = (double)p_detection->bbox.x;
        double                bbox_y      = (double)p_detection->bbox.y;
        double                bbox_w      = (double)p_detection->bbox.width;
        double                bbox_h      = (double)p_detection->bbox.height;

        uint64_t start_time_ns  = (uint64_t)(ns_since_epoch + (uint64_t)(bbox_x * ns_per_pixel));
        double   end_time_ns    = (double)(start_time_ns + (bbox_w * ns_per_pixel));
        double   duration_ns    = (double)(end_time_ns - start_time_ns);
        uint64_t center_freq_hz = (uint64_t)(min_freq_hz + ((bbox_y + (bbox_h / 2.0)) * hz_per_pixel));
        uint64_t bandwidth_hz   = (uint64_t)(bbox_h * hz_per_pixel);
        float    rssi           = p_detection->rssi;

        // update the list of database detections
        mongodb_sink::detection db_detection = { .start_time_ns  = start_time_ns,  .duration_ns  = duration_ns,
                                                 .center_freq_hz = center_freq_hz, .bandwidth_hz = bandwidth_hz,
                                                 .rssi           = rssi };
        db_detections.push_back(db_detection);
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// process a detection group [stream processing]
void engine::process_detection_group_stream( engine::detection_group *p_group )
{
    // note: when stream processing, each detection group only contains a single detections tuple/entry

    std::vector<tye_types::detection> detections       = std::get<0>(p_group->detection_data.at(0));
    tye_types::detections_aux         detections_aux   = std::get<1>(p_group->detection_data.at(0));
    cv::Mat                           detections_image = std::get<2>(p_group->detection_data.at(0));
    tye_spectrogram                  *p_spectrogram    = std::get<3>(p_group->detection_data.at(0));

    // generate detections and insert into the database
    if ( ! this_database_off )
    {
        std::vector<mongodb_sink::detection> db_detections = std::vector<mongodb_sink::detection>();

        this->generate_db_detections(detections_aux, detections, db_detections);
        this_p_mongodb_sink->insert(detections_aux, db_detections);
    }

    // add detections image and spectrogram to the batch
    bool add_to_batch = false;

    if ( this_is_sw_gl_renderer ) // software-based opengl renderer ?? decimate batch rate
    {
        if ( (detections_aux.group_id % 4) == 0 ) { add_to_batch = true; }
    }
    else /* have a real display GPU */ { add_to_batch = true; }

    if ( add_to_batch )
    {
        // within batch, keep in order
        uint64_t sequ_num      = detections_aux.group_id;
        uint32_t num_sequ_nums = (uint32_t)this_detections_batch.sequ_nums.size();
        bool     inserted      = false;

        for ( uint32_t i = 0; i < num_sequ_nums; i++ )
        {
            if ( this_detections_batch.sequ_nums.at(i) > sequ_num )
            {
                std::vector<uint64_t>::iterator sn_pos = (this_detections_batch.sequ_nums.begin() + i);
                this_detections_batch.sequ_nums.insert(sn_pos, sequ_num);

                std::vector<cv::Mat>::iterator img_pos = (this_detections_batch.images.begin() + i);
                this_detections_batch.images.insert(img_pos, detections_image);

                std::vector<tye_spectrogram *>::iterator sg_pos = (this_detections_batch.spectrograms.begin() + i);
                this_detections_batch.spectrograms.insert(sg_pos, p_spectrogram);

                inserted = true;
                break;
            }
        }

        if ( ! inserted )
        {
            this_detections_batch.sequ_nums.push_back(sequ_num);
            this_detections_batch.images.push_back(detections_image);
            this_detections_batch.spectrograms.push_back(p_spectrogram);
        }
    }
    else // release resources
    {
        detections_image.release();
        p_spectrogram->release();
    }

    // have a complete detections batch ??
    uint32_t detections_batch_size = 8;
    if ( this_p_multiplot != nullptr ) { detections_batch_size = this_p_multiplot->get_batch_size(); }

    if ( this_detections_batch.images.size() == detections_batch_size )
    {
        // for each set of adjacent spectrograms, compute the standard-error mean difference
        uint32_t num_images       = (uint32_t)this_detections_batch.images.size();
        uint32_t num_spectrograms = (uint32_t)this_detections_batch.spectrograms.size();
        float    mean_diff        = 0.0f;

        for ( uint32_t i = 0, j = 1; j < num_spectrograms; i++, j++ )
        {
            float *p_spectrogram_1 = this_detections_batch.spectrograms.at(i)->buffer();
            float *p_spectrogram_2 = this_detections_batch.spectrograms.at(j)->buffer();

            mean_diff = this_p_fft_bins_proc->compute_stderr_mean_diff(p_spectrogram_1, p_spectrogram_2, this_stream);
            this_detections_batch.spectrogram_diffs.push_back(mean_diff);

            if ( ! this_database_off )
            {
                this_p_mongodb_sink->update(this_detections_batch.sequ_nums.at(j), mean_diff);
            }
        }

        // update the plot
        if ( this_p_multiplot != nullptr )
        {
            this_p_multiplot->update(this_detections_batch.images, this_detections_batch.spectrogram_diffs);
        }

        // clean up and reset the batch
        //
        // note: for images and spectrograms, keep the last entry around to be the start of the next batch
        for ( uint32_t i = 0; i < (num_images       - 1); i++ ) { this_detections_batch.images.at(i).release();        }
        for ( uint32_t i = 0; i < (num_spectrograms - 1); i++ ) { this_detections_batch.spectrograms.at(i)->release(); }

        std::vector<uint64_t>::iterator sequ_nums_start_pos = this_detections_batch.sequ_nums.begin();
        std::vector<uint64_t>::iterator sequ_nums_end_pos   = this_detections_batch.sequ_nums.end() - 1;
        this_detections_batch.sequ_nums.erase(sequ_nums_start_pos, sequ_nums_end_pos);

        std::vector<cv::Mat>::iterator images_start_pos = this_detections_batch.images.begin();
        std::vector<cv::Mat>::iterator images_end_pos   = this_detections_batch.images.end() - 1;
        this_detections_batch.images.erase(images_start_pos, images_end_pos);

        std::vector<tye_spectrogram *>::iterator sg_start_pos = this_detections_batch.spectrograms.begin();
        std::vector<tye_spectrogram *>::iterator sg_end_pos   = this_detections_batch.spectrograms.end() - 1;
        this_detections_batch.spectrograms.erase(sg_start_pos, sg_end_pos);

        this_detections_batch.spectrogram_diffs.clear();
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// process a detection group [file processing]
void engine::process_detection_group_file( engine::detection_group *p_group )
{
    // note: when file processing, each detection group contains a tuple/entry for each [fft-size x fft-size]
    //       chunk of the file

    uint32_t num_group_detection_tuples = (uint32_t)p_group->detection_data.size();

    for ( uint32_t i = 0; i < num_group_detection_tuples; i++ )
    {
        std::vector<tye_types::detection> detections       = std::get<0>(p_group->detection_data.at(i));
        tye_types::detections_aux         detections_aux   = std::get<1>(p_group->detection_data.at(i));
        cv::Mat                           detections_image = std::get<2>(p_group->detection_data.at(i));
        tye_spectrogram                  *p_spectrogram    = std::get<3>(p_group->detection_data.at(i));

        // generate detections and insert into the database
        if ( ! this_database_off )
        {
            std::vector<mongodb_sink::detection> db_detections = std::vector<mongodb_sink::detection>();

            this->generate_db_detections(detections_aux, detections, db_detections);
            this_p_mongodb_sink->insert(detections_aux, db_detections);
        }

        // if the path is set
        std::string detections_image_path = p_group->image_path;

        if ( ! detections_image_path.empty() )
        {
            // create the path if it does not exist
            if ( ! std::filesystem::exists(detections_image_path) )
            {
                std::filesystem::create_directory(detections_image_path);
            }

            // build the full file path and write the file
            if ( std::filesystem::exists(detections_image_path) )
            {
                std::string full_path = detections_image_path + "/" + std::to_string(detections_aux.sequ_num) + ".png";
                cv::imwrite(full_path, detections_image);
            }
        }

        // release resources
        detections_image.release();
        p_spectrogram->release();
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// clean-up detections groups (release resources, etc)
void engine::cleanup_detection_groups( void )
{
    // clean-up detections batch...release images and spectrograms that are batched for processing
    uint32_t num_images       = (uint32_t)this_detections_batch.images.size();
    uint32_t num_spectrograms = (uint32_t)this_detections_batch.spectrograms.size();

    for ( uint32_t i = 0; i < num_images;       i++ ) { this_detections_batch.images.at(i).release();        }
    for ( uint32_t i = 0; i < num_spectrograms; i++ ) { this_detections_batch.spectrograms.at(i)->release(); }

    this_detections_batch.sequ_nums.clear();
    this_detections_batch.images.clear();
    this_detections_batch.spectrograms.clear();
    this_detections_batch.spectrogram_diffs.clear();

    // release pending detection groups
    engine::detection_group *p_group = nullptr;

    while ( true )
    {
        this_detections_out_mutex.lock();

        if ( this_detection_groups_out.size() > 0 )
        {
            p_group = this_detection_groups_out.at(0);
            this_detection_groups_out.erase(this_detection_groups_out.begin());
        }
        this_detections_out_mutex.unlock();

        if ( p_group == nullptr ) { break; }
        else
        {
            uint32_t num_detections = (uint32_t)p_group->detection_data.size();

            for ( uint32_t i = 0; i < num_detections; i++ )
            {
                cv::Mat          detections_image = std::get<2>(p_group->detection_data.at(i));
                tye_spectrogram *p_spectrogram    = std::get<3>(p_group->detection_data.at(i));

                detections_image.release();
                p_spectrogram->release();
            }
        }
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// engine advertisement thread
void engine::ad_thread( void )
{
    int32_t            ad_socket = -1;
    struct sockaddr_in ad_sa_in  = {};
    int32_t            bcast_on  =  1;
    int32_t            status    = -1;

    this_ad_ready = true;

    // setup the advertisement socket...best effort
    ad_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if ( ad_socket > 0 )
    {
        fcntl(ad_socket, F_SETFL, fcntl(ad_socket, F_GETFL, 0) | O_NONBLOCK); // non-blocking

        status = setsockopt(ad_socket, SOL_SOCKET, SO_BROADCAST, &bcast_on, sizeof(bcast_on));
        if ( status == 0 )
        {
            std::memset(&ad_sa_in, 0, sizeof(ad_sa_in));

            ad_sa_in.sin_family      = AF_INET;
            ad_sa_in.sin_port        = htons(this_p_config->ad_port());
            ad_sa_in.sin_addr.s_addr = htonl(INADDR_BROADCAST);

        }
        else { close(ad_socket); ad_socket = -1; }
    }

    // run until exit is requested
    while ( true )
    {
        // exit requested ??
        if ( this_exit ) { break; }

        if ( ad_socket > 0 )
        {
            // generate advertisement JSON
            rapidjson::Document rj_ad = {};
            rj_ad.SetObject();

            rapidjson::Document::AllocatorType &rj_allocator = rj_ad.GetAllocator();

            rj_ad.AddMember("msg_type",    "ad",                         rj_allocator);
            rj_ad.AddMember("retune_port", this_p_config->retune_port(), rj_allocator);
            rj_ad.AddMember("cnc_port",    this_p_config->cnc_port(),    rj_allocator);

            if ( ! this_database_off )
            {
                rapidjson::Value db_connect_uri(this_p_mongodb_sink->get_connect_uri().c_str(),    rj_allocator);
                rapidjson::Value db_name(this_p_mongodb_sink->get_name().c_str(),                  rj_allocator);
                rapidjson::Value db_collection(this_p_mongodb_sink->get_collection_name().c_str(), rj_allocator);

                rj_ad.AddMember("db_connect_uri", db_connect_uri, rj_allocator);
                rj_ad.AddMember("db_name",        db_name,        rj_allocator);
                rj_ad.AddMember("db_collection",  db_collection,  rj_allocator);
            }

            rapidjson::StringBuffer rj_ad_buffer = {};
            rapidjson::Writer<rapidjson::StringBuffer> rj_ad_writer(rj_ad_buffer);

            rj_ad.Accept(rj_ad_writer);

            // send advertisement
            sendto(ad_socket, rj_ad_buffer.GetString(), strlen(rj_ad_buffer.GetString()), 0,
                   (const struct sockaddr *)&ad_sa_in, sizeof(ad_sa_in));
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    if ( ad_socket > 0 ) { close(ad_socket); }

    this_running = false;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// engine detections processor thread
void engine::dproc_thread( void )
{
    // if processing a stream and a plot was requested
    if ( (this_process_stream) && (this_p_config->boxes_plot_on() || this_p_config->history_plot_on()) )
    {
        // create the multi-plot [best effort]
        this_p_multiplot = new gl_multiplot(this_p_config->boxes_plot_on(), this_p_config->history_plot_on(),
                                            /*image_w*/this_p_config->fft_size(), /*image_h*/this_p_config->fft_size());
        if ( this_p_multiplot != nullptr )
        {
            bool ok = this_p_multiplot->create();
            if ( ok )
            {
                // software renderer ??
                this_gl_renderer_name = this_p_multiplot->get_renderer_name();
                if ( this_gl_renderer_name.find(std::string("llvmpipe")) != std::string::npos )
                {
                    this_is_sw_gl_renderer = true;
                }

/* force HW */  this_is_sw_gl_renderer = false;
            }
            else
            {
                delete this_p_multiplot;
                this_p_multiplot = nullptr;
            }
        }
    }

    this_dproc_ready = true;

    engine::detection_group *p_group         = nullptr;
    uint64_t                 dgrp_proc_start = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::
                                                              system_clock::now().time_since_epoch()).count();
    uint64_t                 dgrp_proc_now   = 0;
    uint32_t                 dgrp_proc_cnt   = 0;
    bool                     dgrp_proc       = false;

    while ( true )
    {
        // exit requested ?? only bail after all detection groups have been processed
        if ( (this_exit) && (! this->detection_groups_to_process()) ) { break; }

        // check if there is a detection group to process
        this_detections_out_mutex.lock();

        if ( this_detection_groups_out.size() > 0 )
        {
            p_group = this_detection_groups_out.at(0);
            this_detection_groups_out.erase(this_detection_groups_out.begin());
        }
        this_detections_out_mutex.unlock();

        // if there is a detection group to process
        if ( p_group != nullptr )
        {
            if ( this_process_stream ) { this->process_detection_group_stream(p_group); }
            else  /* process file */   { this->process_detection_group_file(p_group);   }

            dgrp_proc_cnt++;
            dgrp_proc = true;

            delete p_group;
            p_group = nullptr;
        }
        else { dgrp_proc = false; }

        // did the multi-plot exit ??
        if ( this_p_multiplot != nullptr )
        {
            if ( this_p_multiplot->check_exit() ) { this_exit_cb(); }
        }

#ifdef ENGINE_PROCESS_STREAM_IMAGES_PER_SEC

        // if processing a stream
        if ( this_process_stream )
        {
            // display the number of frames per second
            dgrp_proc_now = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::
                                           system_clock::now().time_since_epoch()).count();

            if ( (dgrp_proc_now - dgrp_proc_start) >= 1000/*millis*/ )
            {
                std::cout << ">> " << engine::NAME << " => IMAGES/SEC [" << dgrp_proc_cnt << "]"
                          << std::endl << std::flush;

                dgrp_proc_start = dgrp_proc_now;
                dgrp_proc_cnt   = 0;
            }
        }

#endif // ENGINE_PROCESS_STREAM_IMAGES_PER_SEC

        std::this_thread::yield();
    }

    // clean up
    this->cleanup_detection_groups();
    if ( this_p_multiplot != nullptr ) { this_p_multiplot->destroy(); }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// engine manager thread
void engine::mgr_thread( void )
{
    this_running      = true;
    this_engine_ready = true;

    // run until exit is requested
    while ( true )
    {
        // exit requested ??
        if ( this_exit ) { break; }

        if ( this_process_stream ) { this->handle_detection_groups_done_stream(); }
        else
        {
            this->handle_detection_groups_done_file();
            this->check_file_processing_done();
        }

        this->handle_cnc();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    this_running = false;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
