//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "tye_app_includes.h"

//-- constants -------------------------------------------------------------------------------------------------------------

// default values
static const uint16_t THIS_DFLT_AD_PORT = 61111;

//-- variables -------------------------------------------------------------------------------------------------------------

// command line arguments
static uint16_t this_ad_port        = THIS_DFLT_AD_PORT;
static float    this_power_offset   = std::numeric_limits<float>::max();
static float    this_pixel_min_val  = std::numeric_limits<float>::max();
static float    this_pixel_max_val  = std::numeric_limits<float>::max();
static uint32_t this_false_detect_w = std::numeric_limits<uint32_t>::max();
static uint32_t this_false_detect_h = std::numeric_limits<uint32_t>::max();

//-- support functions -----------------------------------------------------------------------------------------------------

// display help information
static void display_help( void )
{
    std::cout << std::endl << std::flush;
    std::cout << "----------------------------------------------------------------------------------------------------";
    std::cout << std::endl << std::flush;
    std::cout << "USAGE: ./tye_sp_ad_update_params [ARGS]" << std::endl << std::flush;
    std::cout << "----------------------------------------------------------------------------------------------------";
    std::cout << std::endl << std::flush;
    std::cout << "  --ad-port .......... [OPTIONAL] Advertisement port (default is " << THIS_DFLT_AD_PORT << ")";
    std::cout << std::endl << std::flush;
    std::cout << "  --power-offset ..... [REQUIRED] Power offset value";
    std::cout << std::endl << std::flush;
    std::cout << "  --pixel-min-val .... [REQUIRED] Pixel minimum value";
    std::cout << std::endl << std::flush;
    std::cout << "  --pixel-max-val .... [REQUIRED] Pixel maximum value";
    std::cout << std::endl << std::flush;
    std::cout << "  --false-detect-w ... [REQUIRED] False detection width, in pixels";
    std::cout << std::endl << std::flush;
    std::cout << "  --false-detect-h ... [REQUIRED] False detection height, in pixels";
    std::cout << std::endl << std::flush;
    std::cout << "  --help ............. Display help";
    std::cout << std::endl << std::flush;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// parse the command line
static bool parse_cmdline_args( int32_t num_args, char *p_arg[] )
{
    struct option opt_list[] =
    {
        { "ad-port",        required_argument, nullptr, 0 },
        { "power-offset",   required_argument, nullptr, 1 },
        { "pixel-min-val",  required_argument, nullptr, 2 },
        { "pixel-max-val",  required_argument, nullptr, 3 },
        { "false-detect-w", required_argument, nullptr, 4 },
        { "false-detect-h", required_argument, nullptr, 5 },
        { "help",           no_argument,       nullptr, 6 },
        {  0,               0,                 0,       0 }
    };

    int32_t     opt     = -1;
    int32_t     opt_idx = -1;
    std::string gpus    = std::string("");

    // parse the command line
    while ( true )
    {
        opt = getopt_long(num_args, p_arg, "0123456", opt_list, &opt_idx);
        if ( opt == -1 ) { break; }

        switch ( opt )
        {
            case 0: // ad-port

                this_ad_port = (uint16_t)strtoul(optarg, nullptr, /*base*/10);
                if ( this_ad_port == 0 ) { goto DISPLAY_HELP; }
                break;

            case 1: // power-offset

                this_power_offset = std::stof(std::string(optarg));
                break;

            case 2: // pixel-min-val

                this_pixel_min_val = std::stof(std::string(optarg));
                break;

            case 3: // pixel-max-val

                this_pixel_max_val = std::stof(std::string(optarg));
                break;

            case 4: // false-detect-w

                this_false_detect_w = (uint32_t)strtoul(optarg, nullptr, /*base*/10);
                break;

            case 5: // false-detect-h

                this_false_detect_h = (uint32_t)strtoul(optarg, nullptr, /*base*/10);
                break;

            case 6: // help
            default:

                goto DISPLAY_HELP;
        }
    }

    // verify that all required arguments were provided
    if ( this_power_offset == std::numeric_limits<float>::max() )
    {
        std::cout << std::endl << "ARG => --power-offset REQUIRED" << std::endl << std::flush;
        goto DISPLAY_HELP;
    }

    if ( this_pixel_min_val == std::numeric_limits<float>::max() )
    {
        std::cout << std::endl << "ARG => --pixel-min-val REQUIRED" << std::endl << std::flush;
        goto DISPLAY_HELP;
    }

    if ( this_pixel_max_val == std::numeric_limits<float>::max() )
    {
        std::cout << std::endl << "ARG => --pixel-max-val REQUIRED" << std::endl << std::flush;
        goto DISPLAY_HELP;
    }

    if ( this_false_detect_w == std::numeric_limits<uint32_t>::max() )
    {
        std::cout << std::endl << "ARG => --false-detect-w REQUIRED" << std::endl << std::flush;
        goto DISPLAY_HELP;
    }

    if ( this_false_detect_h == std::numeric_limits<uint32_t>::max() )
    {
        std::cout << std::endl << "ARG => --false-detect-h REQUIRED" << std::endl << std::flush;
        goto DISPLAY_HELP;
    }

    return ( true );

DISPLAY_HELP:
    display_help();

    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// receive and advertisement
static bool recv_ad( uint32_t *p_dst_ipaddr, uint16_t *p_dst_port )
{
    int32_t            ad_socket    = -1;
    struct sockaddr_in bind_sa_in   = {};
    int32_t            reuseaddr_on = 1;
    time_t             timeout_sec  = 0;
    bool               received_ad  = true;
    int32_t            status       = -1;

    std::cout << ">> OPENING AD SOCKET " << std::flush;

    ad_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if ( ad_socket == -1 ) { goto FAILED; }

    fcntl(ad_socket, F_SETFL, fcntl(ad_socket, F_GETFL, 0) | O_NONBLOCK); // non-blocking

    status = setsockopt(ad_socket, SOL_SOCKET, SO_REUSEADDR, &reuseaddr_on, sizeof(reuseaddr_on));
    if ( status == -1 ) { goto FAILED_CLOSE_SOCKET; }

    std::cout << "[OK]" << std::endl << std::flush;
    std::cout << ">> WAITING FOR AD " << std::flush;

    std::memset(&bind_sa_in, 0, sizeof(bind_sa_in));

    bind_sa_in.sin_family      = AF_INET;
    bind_sa_in.sin_addr.s_addr = htonl(INADDR_ANY);
    bind_sa_in.sin_port        = htons(this_ad_port);

    status = bind(ad_socket, (struct sockaddr *)&bind_sa_in, sizeof(bind_sa_in));
    if ( status == -1 ) { goto FAILED_CLOSE_SOCKET; }

    timeout_sec = (time(nullptr) + 2/*seconds*/);

    while ( true )
    {
        char               ad[1024]       = {};
        struct sockaddr_in peer_sa_in     = {};
        socklen_t          peer_sa_in_len = sizeof(peer_sa_in);

        std::memset(&peer_sa_in, 0, sizeof(peer_sa_in));

        ssize_t bytes_recvd = recvfrom(ad_socket, ad, sizeof(ad), 0, (struct sockaddr *)&peer_sa_in, &peer_sa_in_len);
        if ( bytes_recvd > 0 )
        {
            rapidjson::Document rj_msg;
            rj_msg.Parse(ad);

            if ( rj_msg.HasParseError() )
            {
                std::cout << "[PARSE ERROR] [FAIL]" << std::endl << std::flush;
                goto FAILED;
            }

            if ( ! rj_msg.IsObject() )
            {
                std::cout << "[MSG IS NOT AN OBJECT] [FAIL]" << std::endl << std::flush;
                goto FAILED;
            }

            if ( ! rj_msg.HasMember("msg_type") ) { continue; }

            std::string msg_type = rj_msg["msg_type"].GetString();
            if ( msg_type.compare("ad") != 0 ) { continue; }

            if ( ! rj_msg.HasMember("cnc_port") )
            {
                std::cout << "[KEY \"cnc_port\" NOT FOUND] [FAIL]" << std::endl << std::flush;
                goto FAILED;
            }

            if ( ! rj_msg["cnc_port"].IsUint() )
            {
                std::cout << "[KEY \"cnc_port\" INCORRECT TYPE] [FAIL]" << std::endl << std::flush;
                goto FAILED;
            }

            std::cout << "[OK] => " << ad << std::endl << std::flush;

           *p_dst_ipaddr = peer_sa_in.sin_addr.s_addr;
           *p_dst_port   = (uint16_t)rj_msg["cnc_port"].GetUint();
            break;
        }

        if ( time(nullptr) > timeout_sec )
        {
            std::cout << "[TIMEOUT]" << std::endl << std::flush;
            received_ad = false;
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    close(ad_socket);

    return ( received_ad );

FAILED_CLOSE_SOCKET:
    close(ad_socket);

FAILED:
   *p_dst_ipaddr = 0;
   *p_dst_port   = 0;

    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// send an update parameters message and wait for a status/reply
static bool send_update_params_msg_wait_status( uint32_t dst_ipaddr, uint16_t dst_port )
{
    int32_t            cnc_socket = -1;
    struct sockaddr_in dst_sa_in  = {};
    struct in_addr     dst_ip     = { .s_addr = dst_ipaddr };
    std::string        json_msg   = std::string("");
    ssize_t            bytes_sent = 0;

    std::cout << ">> OPENING CNC SOCKET " << std::flush;

    cnc_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if ( cnc_socket == -1 ) { goto FAILED; }

    fcntl(cnc_socket, F_SETFL, fcntl(cnc_socket, F_GETFL, 0) | O_NONBLOCK); // non-blocking
    std::cout << "[OK]" << std::endl << std::flush;

    // build and send an update parameters message
    std::cout << ">> BUILDING UPDATE PARAMS MSG " << std::flush;

    {
        rapidjson::Document rj_msg = {};
        rj_msg.SetObject();

        rapidjson::Document::AllocatorType &rj_allocator = rj_msg.GetAllocator();

        rj_msg.AddMember("msg_type",       "update_params",     rj_allocator);
        rj_msg.AddMember("power_offset",   this_power_offset,   rj_allocator);
        rj_msg.AddMember("pixel_min_val",  this_pixel_min_val,  rj_allocator);
        rj_msg.AddMember("pixel_max_val",  this_pixel_max_val,  rj_allocator);
        rj_msg.AddMember("false_detect_w", this_false_detect_w, rj_allocator);
        rj_msg.AddMember("false_detect_h", this_false_detect_h, rj_allocator);

        rapidjson::StringBuffer rj_msg_buffer = {};
        rapidjson::Writer<rapidjson::StringBuffer> rj_msg_writer(rj_msg_buffer);

        rj_msg_writer.SetMaxDecimalPlaces(6);
        rj_msg.Accept(rj_msg_writer);

        json_msg = rj_msg_buffer.GetString();
    }

    std::cout << "[OK]" << std::endl << std::flush;
    std::cout << ">> SENDING UPDATE PARAMS MSG " << std::flush;

    std::memset(&dst_sa_in, 0, sizeof(dst_sa_in));

    dst_sa_in.sin_family      = AF_INET;
    dst_sa_in.sin_port        = htons(dst_port);
    dst_sa_in.sin_addr.s_addr = dst_ipaddr;

    bytes_sent = sendto(cnc_socket, json_msg.c_str(), json_msg.size(), 0, (struct sockaddr *)&dst_sa_in,
                        sizeof(dst_sa_in));
    if ( bytes_sent != json_msg.size() ) { goto FAILED; }

    std::cout << "[OK] DST [" << inet_ntoa(dst_ip) << ":" << dst_port << "] MSG " << json_msg
              << std::endl << std::flush;

    {
        // wait, with timeout, for update parameters status
        char    status[256] = {};
        time_t  timeout_sec = (time(nullptr) + 2/*seconds*/);
        ssize_t bytes_recvd =  0;

        std::cout << ">> WAITING FOR UPDATE PARAMS STATUS " << std::flush;

        while ( true )
        {
            bytes_recvd = recvfrom(cnc_socket, status, sizeof(status), 0, nullptr, nullptr);
            if ( bytes_recvd > 0 )
            {
                rapidjson::Document rj_status;
                rj_status.Parse(status);

                // check for errors and verify all required keys/values are present
                if ( rj_status.HasParseError() )
                {
                    std::cout << "[PARSE ERROR] [FAIL]" << std::endl << std::flush;
                    break;
                }

                if ( ! rj_status.IsObject() )
                {
                    std::cout << "[MSG IS NOT AN OBJECT] [FAIL]" << std::endl << std::flush;
                    break;
                }

                if ( ! rj_status.HasMember("msg_type") )
                {
                    std::cout << "[KEY \"msg_type\" NOT FOUND] [FAIL]" << std::endl << std::flush;
                    break;
                }

                if ( ! rj_status.HasMember("status") )
                {
                    std::cout << "[KEY \"status\" NOT FOUND] [FAIL]" << std::endl << std::flush;
                    break;
                }

                // extract values
                std::string msg_type = rj_status["msg_type"].GetString();
                std::string status   = rj_status["status"].GetString();

                // is message type update parameters status ??
                if ( msg_type.compare("update_params_status") != 0 )
                {
                    std::cout << "[MSG TYPE INCORRECT] [FAIL]" << std::endl << std::flush;
                    break;
                }

                if ( status.compare("success") == 0 ) { std::cout << "[SUCCESS]" << std::endl; }
                else                                  { std::cout << "[FAIL]"    << std::endl; }

                break;
            }

            if ( time(nullptr) > timeout_sec )
            {
                std::cout << "[TIMEOUT]" << std::endl << std::flush;
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    close(cnc_socket);

    return ( true );

FAILED_CLOSE_SOCKET:
    close(cnc_socket);

FAILED:
    std::cout << "[FAIL]" << std::endl << std::flush;

    return ( false );
}

//-- entry point -----------------------------------------------------------------------------------------------------------

int32_t main( int32_t num_args, char *p_args[] )
{
    uint32_t dst_ipaddr = 0;
    uint16_t dst_port   = 0;
    bool     ok         = false;

    ok = parse_cmdline_args(num_args, p_args);
    if ( ! ok ) { goto FAILED; }

    std::cout << std::endl << std::flush;

    ok = recv_ad(&dst_ipaddr, &dst_port);
    if ( ! ok ) { goto FAILED; }

    ok = send_update_params_msg_wait_status(dst_ipaddr, dst_port);
    if ( ! ok ) { goto FAILED; }

    std::cout << std::endl << std::flush;

    return ( 0 );

FAILED:
    std::cout << std::endl << std::flush;

    return ( -1 );
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
