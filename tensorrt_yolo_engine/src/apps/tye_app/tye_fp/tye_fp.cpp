//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "tye_app_includes.h"
#include "tye_buffer_pool.h"
#include "fp_config.h"
#include "fp_frontend.h"
#include "engine.h"
#include "mongodb_sink.h"

//-- constants -------------------------------------------------------------------------------------------------------------

// fft size and image processing thresholds
static const uint32_t THIS_FFT_SIZE        = 1024;
static const float    THIS_SCORE_THRESHOLD = 0.01f;
static const float    THIS_NMS_THRESHOLD   = 0.1f;

// default values for optional command line arguments
static const uint16_t    THIS_DFLT_AD_PORT        =  61111;
static const uint16_t    THIS_DFLT_CNC_PORT       =  62222;
static const uint16_t    THIS_DFLT_RETUNE_PORT    =  63333;
static const std::string THIS_DFLT_DATABASE_CREDS =  std::string("");
static const std::string THIS_DFLT_DATABASE_IP    =  std::string("127.0.0.1");
static const uint16_t    THIS_DFLT_DATABASE_PORT  =  27017;
static const bool        THIS_DFLT_DATABASE_OFF   = false;
static const float       THIS_DFLT_POWER_OFFSET   =  0.0f;
static const float       THIS_DFLT_PIXEL_MIN_VAL  = -60.0f;
static const float       THIS_DFLT_PIXEL_MAX_VAL  =  0.0f;
static const uint32_t    THIS_DFLT_FALSE_DETECT_W =  800;
static const uint32_t    THIS_DFLT_FALSE_DETECT_H =  400;

//-- variables -------------------------------------------------------------------------------------------------------------

// main components
static tye_buffer_pool *this_p_buffer_pool  = nullptr;
static fp_frontend     *this_p_frontend     = nullptr;
static config          *this_p_config       = nullptr;
static engine          *this_p_engine       = nullptr;
static mongodb_sink    *this_p_mongodb_sink = nullptr;

// exit requested ??
static bool this_exit = false;

//-- support functions -----------------------------------------------------------------------------------------------------

// exit handler [CNTRL-C]
static void exit_handler( int32_t signal )
{
    this_exit = true;
    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// exit callback...provided to the engine component, and called when processing is done
static void exit_cb( void )
{
    this_exit = true;
    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// display help information
static void display_help( void )
{
    std::cout << std::endl << std::flush;
    std::cout << "----------------------------------------------------------------------------------------------------";
    std::cout << std::endl << std::flush;
    std::cout << "USAGE: ./tye_fp [ARGS]" << std::endl << std::flush;
    std::cout << "----------------------------------------------------------------------------------------------------";
    std::cout << std::endl << std::endl << std::flush;
    std::cout << "  --gpus .............. Comma-separted list of GPUs to use (ex. --gpus 0,1,3)";
    std::cout << std::endl << std::flush;
    std::cout << "  --engine-path ....... Path to the TENSOR-RT engine to load and run";
    std::cout << std::endl << std::flush;
    std::cout << "  --engines-per-gpu ... Number of TENSOR-RT engines to load and run on each GPU";
    std::cout << std::endl << std::flush;
    std::cout << "  --process-path ...... Path to a file or directory to be processed";
    std::cout << std::endl << std::flush;
    std::cout << "  --output-path ....... Directory path where output will be written";
    std::cout << std::endl << std::flush;
    std::cout << "  --ad-port ........... Port on which parameters will be advertised";
    std::cout << std::endl << std::flush;
    std::cout << "  --cnc-port .......... Listen port for CnC commands";
    std::cout << std::endl << std::flush;
    std::cout << "  --database-creds .... Database authentication credentials (e.g. USERNAME:PASSWORD)";
    std::cout << std::endl << std::flush;
    std::cout << "  --database-ip ....... IP address of the database server";
    std::cout << std::endl << std::flush;
    std::cout << "  --database-port ..... Port on which the database is listening for connections";
    std::cout << std::endl << std::flush;
    std::cout << "  --database-off ...... Disable database operations";
    std::cout << std::endl << std::flush;
    std::cout << "  --power-offset ...... Power offset value";
    std::cout << std::endl << std::flush;
    std::cout << "  --pixel-min-value ... Minimum pixel value when converting a spectrogram to a black-hot image";
    std::cout << std::endl << std::flush;
    std::cout << "  --pixel-max-value ... Maximum pixel value when converting a spectrogram to a black-hot image";
    std::cout << std::endl << std::flush;
    std::cout << "  --false-detect-w .... False detection box width";
    std::cout << std::endl << std::flush;
    std::cout << "  --false-detect-h .... False detection box height";
    std::cout << std::endl << std::flush;
    std::cout << "  --help .............. Display help";
    std::cout << std::endl << std::endl << std::flush;
    std::cout << "----------------------------------------------------------------------------------------------------";
    std::cout << std::endl << std::flush;
    std::cout << "ARGS INFO" << std::endl << std::flush;
    std::cout << "----------------------------------------------------------------------------------------------------";
    std::cout << std::endl << std::endl << std::flush;
    std::cout << "  [FILE PROCESSING]" << std::endl << std::endl << std::flush;
    std::cout << "     REQUIRED" << std::endl << std::flush;
    std::cout << "       => --gpus" << std::endl << std::flush;
    std::cout << "       => --engine-path" << std::endl << std::flush;
    std::cout << "       => --engines-per-gpu" << std::endl << std::flush;
    std::cout << "       => --process-path" << std::endl << std::flush;
    std::cout << "     OPTIONAL" << std::endl << std::flush;
    std::cout << "       => --output-path [default is \"\"]" << std::endl << std::flush;
    std::cout << "       => --database-creds [default is \"\"]" << std::endl << std::flush;
    std::cout << "       => --database-ip [default is " << THIS_DFLT_DATABASE_IP << "]" << std::endl << std::flush;
    std::cout << "       => --database-port [default is " << THIS_DFLT_DATABASE_PORT << "]" << std::endl << std::flush;
    std::cout << "       => --database-off [default is false]" << std::endl << std::flush;
    std::cout << "       => --power-offset [default is " << THIS_DFLT_POWER_OFFSET << "]" << std::endl << std::flush;
    std::cout << "       => --pixel-min-val [default is " << THIS_DFLT_PIXEL_MIN_VAL << "]" << std::endl << std::flush;
    std::cout << "       => --pixel-max-val [default is " << THIS_DFLT_PIXEL_MAX_VAL << "]" << std::endl << std::flush;
    std::cout << "       => --false-detect-w [default is " << THIS_DFLT_FALSE_DETECT_W << "]" << std::endl << std::flush;
    std::cout << "       => --false-detect-h [default is " << THIS_DFLT_FALSE_DETECT_H << "]" << std::endl << std::flush;
    std::cout << std::endl << std::flush;
    std::cout << "----------------------------------------------------------------------------------------------------";
    std::cout << std::endl << std::flush;
    std::cout << "PROCESSING INFO" << std::endl << std::flush;
    std::cout << "----------------------------------------------------------------------------------------------------";
    std::cout << std::endl << std::endl << std::flush;
    std::cout << "  SUPPORTED FILE TYPES ARE" << std::endl << std::flush;
    std::cout << "    => .sigmf-data, .wav" << std::endl << std::flush;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// parse the command line
static bool parse_cmdline_args( int32_t num_args, char *p_arg[] )
{
    struct option opt_list[] =
    {
        { "gpus",            required_argument, nullptr,  0 },
        { "engine-path",     required_argument, nullptr,  1 },
        { "engines-per-gpu", required_argument, nullptr,  2 },
        { "process-path",    required_argument, nullptr,  3 },
        { "output-path",     required_argument, nullptr,  4 },
        { "ad-port",         required_argument, nullptr,  5 },
        { "cnc-port",        required_argument, nullptr,  6 },
        { "database-creds",  required_argument, nullptr,  7 },
        { "database-ip",     required_argument, nullptr,  8 },
        { "database-port",   required_argument, nullptr,  9 },
        { "database-off",    no_argument,       nullptr, 'A'},
        { "power-offset",    required_argument, nullptr, 'B'},
        { "pixel-min-val",   required_argument, nullptr, 'C'},
        { "pixel-max-val",   required_argument, nullptr, 'D'},
        { "false-detect-w",  required_argument, nullptr, 'E'},
        { "false-detect-h",  required_argument, nullptr, 'F'},
        { "help",            no_argument,       nullptr, 'G'},
        {  0,                0,                 0,        0 }
    };

    std::vector<uint32_t> gpus            = std::vector<uint32_t>();
    std::string           gpus_str        = std::string("");
    std::string           engine_path     = std::string("");
    uint32_t              engines_per_gpu = 0;
    std::string           process_path    = std::string("");
    std::string           output_path     = std::string("");
    uint16_t              ad_port         = THIS_DFLT_AD_PORT;
    uint16_t              cnc_port        = THIS_DFLT_CNC_PORT;
    std::string           database_creds  = THIS_DFLT_DATABASE_CREDS;
    std::string           database_ip     = THIS_DFLT_DATABASE_IP;
    uint16_t              database_port   = THIS_DFLT_DATABASE_PORT;
    bool                  database_off    = THIS_DFLT_DATABASE_OFF;
    float                 power_offset    = THIS_DFLT_POWER_OFFSET;
    float                 pixel_min_val   = THIS_DFLT_PIXEL_MIN_VAL;
    float                 pixel_max_val   = THIS_DFLT_PIXEL_MAX_VAL;
    uint32_t              false_detect_w  = THIS_DFLT_FALSE_DETECT_W;
    uint32_t              false_detect_h  = THIS_DFLT_FALSE_DETECT_H;
    int32_t               opt             = -1;
    int32_t               opt_idx         = -1;

    // parse the command line
    while ( true )
    {
        opt = getopt_long(num_args, p_arg, "0123456789ABCDEFG", opt_list, &opt_idx);
        if ( opt == -1 ) { break; }

        switch ( opt )
        {
            case 0: // gpus

                gpus_str = std::string(optarg);
                break;

            case 1: // engine-path

                engine_path = std::string(optarg);
                break;

            case 2: // engines-per-gpu

                engines_per_gpu = (uint32_t)strtoul(optarg, nullptr, /*base*/10);
                if ( engines_per_gpu == 0 ) { goto DISPLAY_HELP; }
                break;

            case 3: // process-path

                process_path = std::string(optarg);
                if ( process_path.empty() ) { goto DISPLAY_HELP; }
                break;

            case 4: // output-path

                output_path = std::string(optarg);
                if ( output_path.empty() ) { goto DISPLAY_HELP; }
                break;

            case 5: // ad-port

                ad_port = (uint16_t)strtoul(optarg, nullptr, /*base*/10);
                if ( ad_port == 0 ) { goto DISPLAY_HELP; }
                break;

            case 6: // cnc-port

                cnc_port = (uint16_t)strtoul(optarg, nullptr, /*base*/10);
                if ( cnc_port == 0 ) { goto DISPLAY_HELP; }
                break;

            case 7: // database-creds

                database_creds = std::string(optarg);
                break;

            case 8: // database-ip

                database_ip = std::string(optarg);
                break;

            case 9: // database-port

                database_port = (uint16_t)strtoul(optarg, nullptr, /*base*/10);
                if ( database_port == 0 ) { goto DISPLAY_HELP; }
                break;

            case 'A': // database-off

                database_off = true;
                break;

            case 'B': // power-offset

                power_offset = std::stof(std::string(optarg));
                break;

            case 'C': // pixel-min-val

                pixel_min_val = std::stof(std::string(optarg));
                break;

            case 'D': // pixel-max-val

                pixel_max_val = std::stof(std::string(optarg));
                break;

            case 'E': // false-detect-w

                false_detect_w = (uint32_t)strtoul(optarg, nullptr, /*base*/10);
                break;

            case 'F': // false-detect-h

                false_detect_h = (uint32_t)strtoul(optarg, nullptr, /*base*/10);
                break;

            case 'G': // help
            default:

                goto DISPLAY_HELP;
        }
    }

    // verify that all required arguments were provided
    if ( gpus_str.empty() )
    {
        std::cout << std::endl << "ARG => --gpus REQUIRED" << std::endl << std::flush;
        goto DISPLAY_HELP;
    }

    if ( engine_path.empty() )
    {
        std::cout << std::endl << "ARG => --engine-path REQUIRED" << std::endl << std::flush;
        goto DISPLAY_HELP;
    }

    if ( engines_per_gpu == 0 )
    {
        std::cout << std::endl << "ARG => --engines-per-gpu REQUIRED" << std::endl << std::flush;
        goto DISPLAY_HELP;
    }

    if ( process_path.empty() )
    {
        std::cout << std::endl << "ARG => --process-path REQUIRED" << std::endl << std::flush;
        goto DISPLAY_HELP;
    }

    {
        // generate the list of GPUs
        std::stringstream gpus_ss(gpus_str);

        while ( gpus_ss.good() )
        {
            std::string gpu_str = std::string("");
            uint32_t    gpu     = 0;

            std::getline(gpus_ss, gpu_str, ',');
            gpu = (uint32_t)strtoul(gpu_str.c_str(), nullptr, /*base*/10);

            gpus.push_back(gpu);
        }

        if ( gpus.size() == 0 ) { goto DISPLAY_HELP; }
    }

    // create configuration
    this_p_config = new fp_config(gpus, engine_path, engines_per_gpu, process_path, output_path, ad_port, cnc_port,
                                  database_creds, database_ip, database_port, database_off, THIS_FFT_SIZE,
                                  THIS_SCORE_THRESHOLD, THIS_NMS_THRESHOLD, power_offset, pixel_min_val, pixel_max_val,
                                  false_detect_w, false_detect_h);
    if ( this_p_config == nullptr ) { goto FAILED; }

    return ( true );

DISPLAY_HELP:
    display_help();

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// check if a path exists
static bool path_exists( const std::string &path ) { return ( access(path.c_str(), F_OK) == 0 ); }

//--------------------------------------------------------------------------------------------------------------------------

// check if a path is a file
static bool is_file( const std::string &path )
{
    struct stat buffer = {};

    if ( ! path_exists(path) ) { return false; }

    return ( (stat(path.c_str(), &buffer) == 0) && S_ISREG(buffer.st_mode) );
}

//-- component operations --------------------------------------------------------------------------------------------------

// create the buffer pool component
static bool create_buffer_pool( uint32_t num_buffers, uint32_t buffer_len )
{
    bool ok = false;

    std::cout << ">> CREATING BUFFER POOL " << std::flush;

    this_p_buffer_pool = new tye_buffer_pool(num_buffers, buffer_len, /*use malloc*/true);
    if ( this_p_buffer_pool == nullptr )
    {
        std::cout << "[FAIL]" << std::endl << std::flush;
        goto FAILED;
    }

    ok = this_p_buffer_pool->create();
    if ( ! ok )
    {
        std::cout << "[FAIL]" << std::endl << std::flush;
        goto FAILED_FREE_BUFFER_POOL;
    }

    std::cout << "[OK]" << std::endl << std::flush;
    std::cout << "   - [NUM BUFFERS] " << num_buffers << " [BUFFER LEN] " << buffer_len << std::endl << std::flush;

    return ( true );

FAILED_FREE_BUFFER_POOL:
    delete this_p_buffer_pool;
    this_p_buffer_pool = nullptr;

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// destroy the buffer pool component
static void destroy_buffer_pool( void )
{
    std::cout << ">> DESTROYING BUFFER POOL " << std::flush;

    this_p_buffer_pool->destroy();
    delete this_p_buffer_pool;

    std::cout << "[OK]" << std::endl << std::flush;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// create the mongodb sink component
static bool create_mongodb_sink( void )
{
    // database off ??
    if ( this_p_config->database_off() ) { return ( true ); }

    // database is on
    std::string db_uri = this_p_config->database_ip() + ":" + std::to_string(this_p_config->database_port());
    bool        ok     = false;

    std::cout << ">> CREATING MONGODB SINK " << std::flush;

    this_p_mongodb_sink = new mongodb_sink(this_p_config->database_creds(), this_p_config->database_ip(),
                                           this_p_config->database_port());
    if ( this_p_mongodb_sink == nullptr )
    {
        std::cout << "[FAIL]" << std::endl << std::flush;
        goto FAILED;
    }

    if ( ! this_p_config->database_creds().empty() ) { db_uri = this_p_config->database_creds() + "@" + db_uri; }

    std::cout << "[OK]" << std::endl << std::flush;
    std::cout << ">> CONNECTING TO MONGODB SINK [" << db_uri << "] " << std::flush;

    ok = this_p_mongodb_sink->connect();
    if ( ! ok )
    {
        std::cout << "[FAIL]" << std::endl << std::flush;
        goto FAILED_FREE_MONGODB_SINK;
    }

    std::cout << "[OK]" << std::endl << std::flush;

    return ( true );

FAILED_FREE_MONGODB_SINK:
    delete this_p_mongodb_sink;
    this_p_mongodb_sink = nullptr;

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// destroy the mongodb sink component
static void destroy_mongodb_sink( void )
{
    if ( ! this_p_config->database_off() )
    {
        std::cout << ">> DESTROYING MONGODB SINK " << std::flush;

        this_p_mongodb_sink->disconnect();
        delete this_p_mongodb_sink;

        std::cout << "[OK]" << std::endl << std::flush;
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// start the engine component
static bool start_engine( void )
{
    std::string display_gpu = std::string("");
    bool        ok          = false;

    std::cout << ">> STARTING ENGINE " << std::flush;

    this_p_engine = new engine(this_p_config, this_p_mongodb_sink);
    if ( this_p_engine == nullptr )
    {
        std::cout << "[FAIL]" << std::endl << std::flush;
        goto FAILED;
    }

    ok = this_p_engine->start(exit_cb);
    if ( ! ok )
    {
        std::cout << "[FAIL]" << std::endl << std::flush;
        goto FAILED_FREE_ENGINE;
    }

    std::cout << "[OK]" << std::endl << std::flush;

    for ( uint32_t i = 0; i < this_p_config->gpus().size(); i++ )
    {
        cudaDeviceProp gpu_properties = {};
        cudaGetDeviceProperties(&gpu_properties, this_p_config->gpus().at(i));

        std::cout << "   - [COMPUTE GPU] " << this_p_config->gpus().at(i) << " [NAME] " << gpu_properties.name
                  << " [NUM ENGINES] " << this_p_config->engines_per_gpu() << std::endl << std::flush;
    }

    display_gpu = this_p_engine->get_gl_renderer_name();
    if ( ! display_gpu.empty() )
    {
        std::cout << "   - [DISPLAY GPU] " << display_gpu << std::endl << std::flush;
    }

    std::cout << "   - [PORTS] [AD] " << this_p_config->ad_port() << " [CNC] " << this_p_config->cnc_port()
              << std::endl << std::flush;
    std::cout << "   - [POWER OFFSET] " << this_p_config->power_offset() << std::endl << std::flush;
    std::cout << "   - [PIXEL RANGE] " << this_p_config->pixel_min_val() << " TO " << this_p_config->pixel_max_val()
              << std::endl << std::flush;
    std::cout << "   - [FALSE DETECT] " << this_p_config->false_detect_w() << "x" << this_p_config->false_detect_h()
              << std::endl << std::flush;

    return ( true );

FAILED_FREE_ENGINE:
    delete this_p_engine;
    this_p_engine = nullptr;

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// shutdown the engine component
static void shutdown_engine( void )
{
    std::cout << ">> SHUTTING DOWN ENGINE " << std::flush;

    this_p_engine->shutdown();
    delete this_p_engine;

    std::cout << "[OK]" << std::endl << std::flush;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// start the frontend component
static bool start_frontend( void )
{
    bool ok = false;

    std::cout << ">> STARTING FRONTEND " << std::flush;

    this_p_frontend = new fp_frontend(this_p_buffer_pool, this_p_config, this_p_engine);
    if ( this_p_frontend == nullptr )
    {
        std::cout << "[FAIL]" << std::endl << std::flush;
        goto FAILED;
    }

    ok = this_p_frontend->start();
    if ( ! ok )
    {
        std::cout << "[FAIL]" << std::endl << std::flush;
        goto FAILED_FREE_FRONTEND;
    }

    std::cout << "[FILE/DIR PROCESSING MODE] [OK]" << std::endl << std::flush;

    return ( true );

FAILED_FREE_FRONTEND:
    delete this_p_frontend;
    this_p_frontend = nullptr;

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// shutdown the frontend component
static void shutdown_frontend( void )
{
    std::cout << ">> SHUTTING DOWN FRONTEND " << std::flush;

    this_p_frontend->shutdown();
    delete this_p_frontend;

    std::cout << "[OK]" << std::endl << std::flush;

    return;
}

//-- entry point -----------------------------------------------------------------------------------------------------------

int32_t main( int32_t num_args, char *p_args[] )
{
    bool ok = parse_cmdline_args(num_args, p_args);
    if ( ! ok ) { goto FAILED; }

    {
        // make sure the engine exist
        if ( ! is_file(this_p_config->engine_path()) )
        {
            std::cout << ">> ENGINE [" << this_p_config->engine_path() << "] DOES NOT EXIST [FAIL]"
                      << std::endl << std::flush;
            goto FAILED;
        }

        std::cout << std::endl << std::flush;

        uint32_t num_buffers = (uint32_t)(this_p_config->gpus().size() * this_p_config->engines_per_gpu() * 8);
        uint32_t buffer_len  = (uint32_t)(THIS_FFT_SIZE * THIS_FFT_SIZE * sizeof(std::complex<float>));

        if ( num_buffers < 32 ) { num_buffers = 32; }

        // create the buffer pool component
        ok = create_buffer_pool(num_buffers, buffer_len);
        if ( ! ok ) { goto FAILED; }

        // create the mongodb sink component
        ok = create_mongodb_sink();
        if ( ! ok ) { goto FAILED_DESTROY_BUFFER_POOL; }

        // start the engine component
        ok = start_engine();
        if ( ! ok ) { goto FAILED_DESTROY_MONGODB_SINK; }

        // start the frontend component
        ok = start_frontend();
        if ( ! ok ) { goto FAILED_SHUTDOWN_ENGINE; }

        // install the exit handler
        signal(SIGINT, exit_handler);

        // run until exit is requested
        while ( true )
        {
            if ( this_exit ) { break; }

            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        std::cout << std::endl << std::flush;

        // clean up
        shutdown_frontend();
        shutdown_engine();
        destroy_mongodb_sink();
        destroy_buffer_pool();

        std::cout << std::endl << std::flush;
    }

    return ( 0 );

FAILED_SHUTDOWN_ENGINE:
    shutdown_engine();

FAILED_DESTROY_MONGODB_SINK:
    destroy_mongodb_sink();

FAILED_DESTROY_BUFFER_POOL:
    destroy_buffer_pool();

FAILED:
    std::cout << std::endl << std::flush;

    return ( -1 );
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
