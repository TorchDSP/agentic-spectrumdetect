//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "sp_frontend.h"

//-- constructor(s) --------------------------------------------------------------------------------------------------------

sp_frontend::sp_frontend( tye_buffer_pool *p_buffer_pool, config *p_config, engine *p_engine )
{
    // initialize
    this_p_mgr_thread    = nullptr;
    this_p_stream_buffer = nullptr;
    this_p_buffer_pool   = p_buffer_pool;
    this_p_config        = p_config;
    this_p_engine        = p_engine;
    this_state           = sp_frontend::state::CONNECT_RADIO;
    this_retune_socket   = -1;
    this_retuned         = false;
    this_group_id        = 0;
    this_sequ_num        = 0;
    this_running         = false;
    this_connected       = false;
    this_exit            = false;

    return;    
}

//-- destructor ------------------------------------------------------------------------------------------------------------

sp_frontend::~sp_frontend( void )
{
    // clean up
    this->shutdown();

    return;
}

//-- public methods --------------------------------------------------------------------------------------------------------

bool sp_frontend::start( void )
{
    // create the radio
    this_p_radio = new sh_sm_radio();
    if ( this_p_radio == nullptr ) { goto FAILED; }

    // start the frontend manager thread
    this_p_mgr_thread = new std::thread(&sp_frontend::mgr_thread, this);
    if ( this_p_mgr_thread == nullptr ) { delete this_p_radio; goto FAILED; }

    return ( true );

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// shutdown frontend
void sp_frontend::shutdown( void )
{
    // clean up
    if ( this_running )
    {
        this_exit = true;
        this_p_mgr_thread->join();
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// check if the frontend is running
bool sp_frontend::is_running( void ) { return ( this_running ); }

//-- private methods --------------------------------------------------------------------------------------------------------

// handle clean up
void sp_frontend::clean_up( void )
{
    if ( this_retune_socket != -1 )
    {
        close(this_retune_socket);
        this_retune_socket = -1;
    }

    if ( this_p_radio != nullptr )
    {
        if ( this_connected )
        {
            this_p_radio->disconnect();
            this_connected = false;
        }

        delete this_p_radio;
        this_p_radio = nullptr;
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// send retune status
void sp_frontend::send_retune_status( bool success, struct sockaddr_in *p_peer_sa_in, ssize_t peer_sa_in_len )
{
    rapidjson::Document rj_msg = {};
    rj_msg.SetObject();

    rapidjson::Document::AllocatorType &rj_allocator = rj_msg.GetAllocator();

    rj_msg.AddMember("msg_type", "retune_status", rj_allocator);
    if ( success ) { rj_msg.AddMember("status", "success", rj_allocator); }
    else           { rj_msg.AddMember("status", "failure", rj_allocator); }

    rapidjson::StringBuffer rj_msg_buffer = {};
    rapidjson::Writer<rapidjson::StringBuffer> rj_msg_writer(rj_msg_buffer);

    rj_msg.Accept(rj_msg_writer);

    sendto(this_retune_socket, rj_msg_buffer.GetString(), strlen(rj_msg_buffer.GetString()), 0,
           (struct sockaddr *)p_peer_sa_in, peer_sa_in_len);

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// handle the CONNECT-RADIO state
void sp_frontend::handle_state_connect_radio( void )
{
    bool ok = false;

    std::cout << ">> " << sp_frontend::NAME << " => CONNECTING TO RADIO " << std::flush;

    ok = this_p_radio->connect_usb();
    if ( ! ok )
    {
        std::cout << "[FAIL]" << std::endl << std::flush;
        goto FAILED;
    }

    std::cout << "[" << this_p_radio->get_device_name() << "] [OK]" << std::endl << std::flush;

    // success...transition state
    this_state     = sp_frontend::state::SETUP_RETUNE;
    this_connected = true;

    return;

FAILED:
    // failed...transition state
    this_state = sp_frontend::state::FAILED;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// handle the SETUP-RETUNE state
void sp_frontend::handle_state_setup_retune( void )
{
    struct sockaddr_in bind_addr    = {};
    int32_t            reuseaddr_on =  1;
    int32_t            status       = -1;

    std::cout << ">> " << sp_frontend::NAME << " => SETTING UP RETUNE " << std::flush;

    // create a UDP socket and bind it to the requested retune port
    this_retune_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if ( this_retune_socket == -1 ) { goto FAILED; }

    fcntl(this_retune_socket, F_SETFL, fcntl(this_retune_socket, F_GETFL, 0) | O_NONBLOCK); // non-blocking

    status = setsockopt(this_retune_socket, SOL_SOCKET, SO_REUSEADDR, &reuseaddr_on, sizeof(reuseaddr_on));
    if ( status == -1 ) { goto FAILED_CLOSE_SOCKET; }

    std::memset(&bind_addr, 0, sizeof(bind_addr));

    bind_addr.sin_family      = AF_INET;
    bind_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    bind_addr.sin_port        = htons(this_p_config->retune_port());

    status = bind(this_retune_socket, (struct sockaddr *)&bind_addr, sizeof(bind_addr));
    if ( status == -1 ) { goto FAILED_CLOSE_SOCKET; }

    // success...transition state
    this_state = sp_frontend::state::CONFIGURE_RADIO;
    std::cout << "[OK] PORT [" << this_p_config->retune_port() << "]" << std::endl << std::flush;

    return;

FAILED_CLOSE_SOCKET:
    close(this_retune_socket);
    this_retune_socket = -1;

FAILED:
    // failed...transition state
    this_state = sp_frontend::state::FAILED;
    std::cout << "[FAIL]" << std::endl << std::flush;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// handle the CONFIGURE-RADIO state
void sp_frontend::handle_state_configure_radio( void )
{
    bool ok = false;

    std::cout << ">> " << sp_frontend::NAME << " => CONFIGURING RADIO " << std::flush;

    ok = this_p_radio->configure(this_p_config->sample_rate_hz(), this_p_config->center_freq_hz(),
                                 this_p_config->atten_db(), this_p_config->ref_level());
    if ( ! ok )
    {
        std::cout << "[FAIL]" << std::endl << std::flush;
        goto FAILED;
    }

    std::cout << "[OK]" << std::endl << std::flush;
    std::cout << ">> " << sp_frontend::NAME << " => RUNNING [OK] SAMPLE RATE [" << this_p_radio->get_sample_rate_mhz()
              << " MHz] BANDWIDTH [" << this_p_radio->get_bandwidth_mhz() << " MHz] CENTER FREQ ["
              << this_p_radio->get_center_freq_mhz() << " MHz]" << std::flush;

    std::cout << " ATTEN [" << std::flush;
    if ( this_p_radio->atten_is_auto() ) { std::cout << std::string("AUTO]");                   }
    else                                 { std::cout << this_p_radio->get_atten_db() << " dB]"; }
    std::cout << std::flush;

    std::cout << " REF LEVEL [" << this_p_radio->get_ref_level() << "]" << std::endl << std::flush;

    // success...transition state
    this_state = sp_frontend::state::GET_RECV_BUFFER;

    return;

FAILED:
    // failed...transition state
    this_state = sp_frontend::state::FAILED;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// handle the GET-RECV-BUFFER state
void sp_frontend::handle_state_get_recv_buffer( void )
{
    // attempt to get an available buffer from the buffer pool
    this_p_stream_buffer = this_p_buffer_pool->get();
    if ( this_p_stream_buffer != nullptr )
    {
        // set buffer attributes
        this_p_stream_buffer->set_source_name(std::string("stream"));
        this_p_stream_buffer->set_group_id(this_group_id++);
        this_p_stream_buffer->set_sequ_num(this_sequ_num);
        this_p_stream_buffer->set_radio_retuned(this_retuned);
        this_p_stream_buffer->set_sample_rate_hz(this_p_radio->get_sample_rate_hz());
        this_p_stream_buffer->set_bandwidth_hz(this_p_radio->get_bandwidth_hz());
        this_p_stream_buffer->set_center_freq_hz(this_p_radio->get_center_freq_hz());
        this_p_stream_buffer->set_atten_db(this_p_radio->get_atten_db());
        this_p_stream_buffer->set_ref_level(this_p_radio->get_ref_level());

        this_retuned = false;

        // success...transition state
        this_state = sp_frontend::state::RECV_SAMPLES;
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// handle the RECV-SAMPLES state
void sp_frontend::handle_state_recv_samples( void )
{
    int64_t ns_since_epoch  = 0;
    bool    samples_dropped = false;
    bool    ok              = false;

    // receive samples into buffer
    ok = this_p_radio->recv_samples(this_p_stream_buffer->get(), this_p_stream_buffer->len(), &ns_since_epoch,
                                    &samples_dropped);
    if ( ! ok ) { goto FAILED; }

    // submit the buffer to the engine for processing
    this_p_stream_buffer->set_samples_ns_since_epoch((uint64_t)ns_since_epoch);
    this_p_engine->process(this_p_stream_buffer);

    if ( samples_dropped ) { std::cout << "D" << std::flush; }

    // success...transition state
    this_state = sp_frontend::state::HANDLE_RETUNE;

    return;

FAILED:
    // failed...transition state
    this_state = sp_frontend::state::FAILED;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// handle the HANDLE-RETUNE state
void sp_frontend::handle_state_handle_retune( void )
{
    char               retune[1024]   = {};
    struct sockaddr_in peer_sa_in     = {};
    socklen_t          peer_sa_in_len = sizeof(peer_sa_in);
    ssize_t            bytes_recvd    = 0;

    std::memset(&peer_sa_in, 0, sizeof(peer_sa_in));

    // attempt to receve a retune command
    bytes_recvd = recvfrom(this_retune_socket, retune, sizeof(retune), 0, (struct sockaddr *)&peer_sa_in,
                           &peer_sa_in_len);
    if ( bytes_recvd > 0 )
    {
        std::cout << ">> " << sp_frontend::NAME << " => RETUNING RADIO " << std::flush;

        rapidjson::Document rj_retune;
        rj_retune.Parse(retune);

        // parse the message and verify all required keys/values are present
        if ( rj_retune.HasParseError() )
        {
            std::cout << "[PARSE ERROR] " << std::flush;
            goto FAILED;
        }

        if ( ! rj_retune.IsObject() )
        {
            std::cout << "[MSG IS NOT AN OBJECT] " << std::flush;
            goto FAILED;
        }

        if ( ! rj_retune.HasMember("msg_type") )
        {
            std::cout << "[KEY \"msg_type\" NOT FOUND] " << std::flush;
            goto FAILED;
        }

        std::string msg_type = rj_retune["msg_type"].GetString();
        if ( msg_type.compare("retune") != 0 )
        {
            std::cout << "[MSG TYPE INCORRECT] " << std::flush;
            goto FAILED;
        }

        if ( ! rj_retune.HasMember("sample_rate_hz") )
        {
            std::cout << "[KEY \"sample_rate_hz\" NOT FOUND] " << std::flush;
            goto FAILED_SEND_STATUS;
        }

        if ( ! rj_retune.HasMember("center_freq_hz") )
        {
            std::cout << "[KEY \"center_freq_hz\" NOT FOUND] " << std::flush;
            goto FAILED_SEND_STATUS;
        }

        if ( ! rj_retune.HasMember("atten_db") )
        {
            std::cout << "[KEY \"atten_db\" NOT FOUND] " << std::flush;
            goto FAILED_SEND_STATUS;
        }

        if ( ! rj_retune.HasMember("ref_level") )
        {
            std::cout << "[KEY \"ref_level\" NOT FOUND] " << std::flush;
            goto FAILED_SEND_STATUS;
        }

        // extract values
        uint64_t sample_rate_hz = (uint64_t)rj_retune["sample_rate_hz"].GetUint64();
        uint64_t center_freq_hz = (uint64_t)rj_retune["center_freq_hz"].GetUint64();
        int32_t  atten_db       = (int32_t)rj_retune["atten_db"].GetInt();
        double   ref_level      = (double)rj_retune["ref_level"].GetDouble();

        // reconfigure the radio
        bool ok = this_p_radio->configure(sample_rate_hz, center_freq_hz, atten_db, ref_level);
        if ( ! ok ) { goto FAILED_SEND_STATUS; }

        // update configuration
        this_p_config->set_sample_rate_hz(this_p_radio->get_sample_rate_hz());
        this_p_config->set_center_freq_hz(this_p_radio->get_center_freq_hz());
        this_p_config->set_atten_db(this_p_radio->get_atten_db());
        this_p_config->set_ref_level(this_p_radio->get_ref_level());

        this_retuned = true;

        std::cout << "[OK] SAMPLE RATE [" << sample_rate_hz << " Hz] CENTER FREQ [" << center_freq_hz << " Hz] ATTEN ["
                  << atten_db << " dB] REF LEVEL [" << ref_level << "]" << std::endl << std::flush;

        // send retune status => success
        this->send_retune_status(true, &peer_sa_in, peer_sa_in_len);
    }

    // transition state
    this_state = sp_frontend::state::GET_RECV_BUFFER;

    return;

FAILED_SEND_STATUS:
    this->send_retune_status(false, &peer_sa_in, peer_sa_in_len);

FAILED:
    std::cout << "[FAIL]" << std::endl << std::flush;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// handle the FAILED state
void sp_frontend::handle_state_failed( void )
{
    this->clean_up();

    // transition state
    this_state = sp_frontend::state::IDLE;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// handle the IDLE state
void sp_frontend::handle_state_idle( void )
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// frontend manager [main] thread
void sp_frontend::mgr_thread( void )
{
    this_running = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    while ( true )
    {
        // exit requested ??
        if ( this_exit ) { break; }

        // handle the current state
        switch ( this_state )
        {
            case sp_frontend::state::CONNECT_RADIO:   this->handle_state_connect_radio();   break;
            case sp_frontend::state::SETUP_RETUNE:    this->handle_state_setup_retune();    break;
            case sp_frontend::state::CONFIGURE_RADIO: this->handle_state_configure_radio(); break;
            case sp_frontend::state::GET_RECV_BUFFER: this->handle_state_get_recv_buffer(); break;
            case sp_frontend::state::RECV_SAMPLES:    this->handle_state_recv_samples();    break;
            case sp_frontend::state::HANDLE_RETUNE:   this->handle_state_handle_retune();   break;
            case sp_frontend::state::FAILED:          this->handle_state_failed();          break;
            case sp_frontend::state::IDLE:            this->handle_state_idle();            break;
            default:                                                                        break;
        }

        std::this_thread::yield();
    }

    this->clean_up();
    this_running = false;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
