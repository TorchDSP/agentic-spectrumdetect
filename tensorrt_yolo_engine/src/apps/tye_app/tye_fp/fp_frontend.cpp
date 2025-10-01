//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "sigmf_file.h"
#include "fp_frontend.h"

//-- constructor(s) --------------------------------------------------------------------------------------------------------

fp_frontend::fp_frontend( tye_buffer_pool *p_buffer_pool, config *p_config, engine *p_engine )
{
    // initialize
    this_p_mgr_thread  = nullptr;
    this_p_buffer_pool = p_buffer_pool;
    this_p_config      = p_config;
    this_p_engine      = p_engine;
    this_path          = std::string("");
    this_process_file  = false;
    this_process_dir   = false;
    this_group_id      = 0;
    this_sequ_num      = 0;
    this_running       = false;
    this_exit          = false;

    return;    
}

//-- destructor ------------------------------------------------------------------------------------------------------------

fp_frontend::~fp_frontend( void )
{
    // clean up
    this->shutdown();

    return;
}

//-- public methods --------------------------------------------------------------------------------------------------------

bool fp_frontend::start( void )
{
    std::string path    = this_p_config->process_path();
    bool        is_file = false;
    bool        is_dir  = false;
    bool        ok      = false;

    // verify that the path exists, and is a file or a directory
    if      ( this->path_is_file(path) ) { is_file = true; }
    else if ( this->path_is_dir(path)  ) { is_dir  = true; }
    else
    {
        std::cout << ">> " << fp_frontend::NAME << " => PATH [" << path << "] IS NOT A FILE OR DIRECTORY [FAIL]"
                  << std::endl << std::flush;
        goto FAILED;
    }

    // if the path is a file, make sure it is a suppored file type
    if ( is_file )
    {
        std::filesystem::path fs_path(path);

        if ( fs_path.extension().string() != ".sigmf-data" )
        {
            std::cout << ">> " << fp_frontend::NAME << " => PATH [" << path << "] IS NOT A SUPPORTED FILE TYPE [FAIL]"
                      << std::endl << std::flush;
            goto FAILED;
        }
    }

    // start the frontend manager thread
    this_p_mgr_thread = new std::thread(&fp_frontend::mgr_thread, this);
    if ( this_p_mgr_thread == nullptr ) { goto FAILED; }

    // set processing information
    this_path         = path;
    this_process_file = is_file;
    this_process_dir  = is_dir;

    return ( true );

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// shutdown frontend
void fp_frontend::shutdown( void )
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
bool fp_frontend::is_running( void ) { return ( this_running ); }

//-- private methods --------------------------------------------------------------------------------------------------------

// check if a path exists
bool fp_frontend::path_exists( std::string &path ) { return ( access(path.c_str(), F_OK) == 0 ); }

//--------------------------------------------------------------------------------------------------------------------------

// check if a path is a file
bool fp_frontend::path_is_file( std::string &path )
{
    struct stat buffer = {};

    if ( ! path_exists(path) ) { return false; }

    return ( (stat(path.c_str(), &buffer) == 0) && S_ISREG(buffer.st_mode) );
}

//--------------------------------------------------------------------------------------------------------------------------

// check if a path is a directory
bool fp_frontend::path_is_dir( std::string &path )
{
    struct stat buffer = {};

    if ( ! path_exists(path) ) { return false; }

    return ( (stat(path.c_str(), &buffer) == 0) && S_ISDIR(buffer.st_mode) );
}

//--------------------------------------------------------------------------------------------------------------------------

// process a loaded file
void fp_frontend::process_loaded_file( file_base *p_file )
{
    tye_buffer *p_buffer = nullptr;
    bool        ok       = true;

    while ( true )
    {
        // attempt to get an available buffer from the buffer pool
        p_buffer = this_p_buffer_pool->get();
        if ( p_buffer == nullptr )
        {
            std::this_thread::yield();
            continue;
        }

        // load the buffer with samples
        ok = p_file->get_samples(p_buffer->get(), p_buffer->len());
        if ( ok )
        {
            // set buffer attributes
            p_buffer->set_source_name(p_file->get_file_path());
            p_buffer->set_source_is_file();
            p_buffer->set_group_id(this_group_id);
            p_buffer->set_sequ_num(this_sequ_num++);
            p_buffer->set_sample_rate_hz(p_file->get_sample_rate());
            p_buffer->set_center_freq_hz(p_file->get_center_freq());

            // submit the buffer to the engine for processing
            this_p_engine->process(p_buffer);
        }
        else
        {
            // let the engine know we have reached end-of-file
            p_buffer->set_source_name(p_file->get_file_path());
            p_buffer->set_source_is_file();
            p_buffer->set_is_eof();

            this_p_engine->process(p_buffer);

            break;
        }
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// process a file
bool fp_frontend::process_file( void )
{
    tye_buffer *p_buffer = nullptr;
    file_base  *p_file   = nullptr;
    bool        ok       = true;

    std::cout << ">> " << fp_frontend::NAME << " => LOADING FILE [" << this_path << "] " << std::flush;

    // open and load the file
    std::filesystem::path fs_path(this_path);

    if ( fs_path.extension().string() == ".sigmf-data" ) { p_file = new sigmf_file(this_path); }

    if ( p_file == nullptr )
    {
        std::cout << "[FAIL]" << std::endl << std::flush;
        goto FAILED;
    }

    ok = p_file->load();
    if ( ! ok )
    {
        std::cout << "[FAIL]" << std::endl << std::flush;
        goto FAILED_FREE_FILE;
    }

    std::cout << "[OK]" << std::endl << std::flush;

    // process the loaded file
    this->process_loaded_file(p_file);
    delete p_file;

    return ( true );

FAILED_FREE_FILE:
    delete p_file;

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// process a directory of files
bool fp_frontend::process_dir( void )
{
    std::vector<std::string> file_paths     = std::vector<std::string>();
    std::vector<file_base *> files          = std::vector<file_base *>();
    uint32_t                 num_file_paths = 0;
    uint32_t                 num_files      = 0;
    bool                     ok             = false;

    // build a list of files (path/name) found in the requested directory
    for ( const auto &dir_entry : std::filesystem::directory_iterator(this_path) )
    {
        if ( std::filesystem::is_regular_file(dir_entry) )
        {
            std::filesystem::path fs_path(dir_entry);

            if ( fs_path.extension().string() == ".sigmf-data" )
            {
                file_paths.push_back(dir_entry.path().string());
            }
        }
    }

    // if no supported files were found in the requested directory
    if ( file_paths.empty() )
    {
        std::cout << ">> " << fp_frontend::NAME << " => NO SUPPORTED FILES FOUND IN DIRECTORY [" << this_path << "]"
                  << std::endl << std::flush;
        goto FAILED;
    }

    // load all of the files that were found in the requested directory
    num_file_paths = (uint32_t)file_paths.size();

    for ( uint32_t i = 0; i < num_file_paths; i++ )
    {
        std::cout << ">> " << fp_frontend::NAME << " => LOADING FILE [" << file_paths.at(i) << "] " << std::flush;

        // open and load the file
        std::filesystem::path fs_path(file_paths.at(i));
        file_base            *p_file = nullptr;

        if ( fs_path.extension().string() == ".sigmf-data" ) { p_file = new sigmf_file(file_paths.at(i)); }

        if ( p_file == nullptr )
        {
            std::cout << "[FAIL]" << std::endl << std::flush;
            goto FAILED;
        }

        ok = p_file->load();
        if ( ! ok )
        {
            std::cout << "[FAIL]" << std::endl << std::flush;

            delete p_file;
            continue;
        }

        files.push_back(p_file);
        std::cout << "[OK]" << std::endl << std::flush;
    }

    // if none of the files were successfully loaded
    if ( files.empty() )
    {
        std::cout << ">> " << fp_frontend::NAME << " => NO FILES LOADED" << std::endl << std::flush;
        goto FAILED;
    }

    // for each file that was successfully loaded
    num_files = (uint32_t)files.size();

    for ( uint32_t i = 0; i < num_files; i++ )
    {
        // process the loaded file
        this->process_loaded_file(files.at(i));
        delete files.at(i);

        // was exit requested ??
        if ( this_exit ) { break; }

        // increment the group id (next file) and reset the sequence number
        this_group_id++;
        this_sequ_num = 0;
    }

    return ( true );

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// frontend manager [main] thread
void fp_frontend::mgr_thread( void )
{
    this_running = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    while ( true )
    {
        // exit requested ??
        if ( this_exit ) { break; }

        // process stream, file, directory
        if      ( this_process_file ) { this->process_file(); }
        else if ( this_process_dir  ) { this->process_dir();  }

        this_process_file = false;
        this_process_dir  = false;

        std::this_thread::yield();
    }

    this_running = false;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
