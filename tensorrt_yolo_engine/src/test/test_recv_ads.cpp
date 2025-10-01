//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "tye_app_includes.h"

//-- variables -------------------------------------------------------------------------------------------------------------

static uint16_t this_recv_port   =  61111;
static int32_t  this_recv_socket = -1;
static bool     this_exit        = false;

//-- support functions -----------------------------------------------------------------------------------------------------

// handle <CNTRL-C>
static void exit_handler( int32_t signal )
{
    if ( signal == SIGINT ) { this_exit = true; }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// display help information
static void display_help( void )
{
    std::cout << std::endl << std::flush;
    std::cout << "----------------------------------------------------------------------------------------------------";
    std::cout << std::endl << std::flush;
    std::cout << "USAGE: ./test_recv_ads [ARGS]" << std::endl << std::flush;
    std::cout << "----------------------------------------------------------------------------------------------------";
    std::cout << std::endl << std::flush;
    std::cout << "  --recv-port ... [OPTIONAL] Port on which to receive detections (default is 61111)";
    std::cout << std::endl << std::flush;
    std::cout << "  --help ........ Display help";
    std::cout << std::endl << std::flush;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// parse the command line
static bool parse_cmdline_args( int32_t num_args, char *p_arg[] )
{
    struct option opt_list[] =
    {
        { "recv-port", required_argument, nullptr, 0 },
        { "help",      no_argument,       nullptr, 1 },
        {  0,          0,                 0,       0 }
    };

    int32_t opt     = -1;
    int32_t opt_idx = -1;

    // parse the command line
    while ( true )
    {
        opt = getopt_long(num_args, p_arg, "01", opt_list, &opt_idx);
        if ( opt == -1 ) { break; }

        switch ( opt )
        {
            case 0: // recv-port

                this_recv_port = (uint16_t)strtoul(optarg, nullptr, /*base*/10);
                if ( this_recv_port == 0 ) { goto DISPLAY_HELP; }
                break;

            case 1: // help
            default:

                goto DISPLAY_HELP;
        }
    }

    return ( true );

DISPLAY_HELP:
    display_help();

    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// setup to receive advertisements
static bool setup_recv( void )
{
    struct sockaddr_in bind_addr    = {};
    int32_t            reuseaddr_on =  1;
    int32_t            status       = -1;

    // create a UDP socket and bind it to the requested port
    this_recv_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if ( this_recv_socket == -1 ) { goto FAILED; }

    fcntl(this_recv_socket, F_SETFL, fcntl(this_recv_socket, F_GETFL, 0) | O_NONBLOCK); // non-blocking

    status = setsockopt(this_recv_socket, SOL_SOCKET, SO_REUSEADDR, &reuseaddr_on, sizeof(reuseaddr_on));
    if ( status == -1 ) { goto FAILED_CLOSE_SOCKET; }

    std::memset(&bind_addr, 0, sizeof(bind_addr));

    bind_addr.sin_family      = AF_INET;
    bind_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    bind_addr.sin_port        = htons(this_recv_port);

    status = bind(this_recv_socket, (struct sockaddr *)&bind_addr, sizeof(bind_addr));
    if ( status == -1 ) { goto FAILED_CLOSE_SOCKET; }

    return ( true );

FAILED_CLOSE_SOCKET:
    close(this_recv_socket);
    this_recv_socket = -1;

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// shutdown receive advertisements
static void shutdown_recv( void )
{
    close(this_recv_socket);
    this_recv_socket = -1;

    return;
}

//-- entry point -----------------------------------------------------------------------------------------------------------

int32_t main( int32_t num_args, char *p_args[] )
{
    char    recv_buffer[1024] = {};
    int32_t bytes_recvd       = 0;
    bool    ok                = false;

    ok = parse_cmdline_args(num_args, p_args);
    if ( ! ok ) { goto FAILED; }

    ok = setup_recv();
    if ( ! ok ) { goto FAILED; }

    std::cout << std::endl << std::flush;
    std::cout << ">> RECV ADS ON PORT " << this_recv_port << std::endl << std::flush;
    std::cout << std::endl << std::flush;

    signal(SIGINT, exit_handler);

    while ( true )
    {
        if ( this_exit ) { break; }

        std::memset(recv_buffer, 0, sizeof(recv_buffer));
        bytes_recvd = recvfrom(this_recv_socket, recv_buffer, sizeof(recv_buffer), 0, nullptr, nullptr);

        if ( bytes_recvd > 0 ) { std::cout << std::string(recv_buffer) << std::endl << std::flush; }
        else                   { std::this_thread::sleep_for(std::chrono::milliseconds(1));        }
    }

    shutdown_recv();
    std::cout << std::endl << std::flush;

    return ( 0 );

FAILED:
    std::cout << std::endl << std::flush;

    return ( -1 );
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
