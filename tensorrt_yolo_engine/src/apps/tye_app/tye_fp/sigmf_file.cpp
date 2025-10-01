//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "sigmf_file.h"

//-- constructor(s) --------------------------------------------------------------------------------------------------------

sigmf_file::sigmf_file( std::string file_path )
    : file_base( file_path )
{
    // initialize
    this_meta_json     = rapidjson::Document();
    this_meta_file     = std::ifstream();
    this_data_file     = std::ifstream();
    this_data_file_len = 0;
    this_sample_rate   = 0;
    this_center_freq   = 0;

    this_signals.clear();

    return;
}

//-- destructor ------------------------------------------------------------------------------------------------------------

sigmf_file::~sigmf_file( void )
{
    // clean up
    if ( this_is_loaded ) { this_data_file.close(); }

    return;
}

//-- public methods --------------------------------------------------------------------------------------------------------

// load a sigmf file
bool sigmf_file::load( void )
{
    std::filesystem::path file_path(this_file_path);

    std::string filename_no_ext = std::string("");
    std::string meta_file_name  = std::string("");
    std::string meta_file_path  = std::string("");
    bool        ok              = false;

    // verify the data file path and open the data file
    ok = this->path_is_file(file_path.string());
    if ( ! ok ) { goto FAILED; }

    this_data_file.open(file_path.string(), (std::ios::binary | std::ios::in));
    if ( ! this_data_file.is_open() ) { goto FAILED; }

    // get the length (in bytes) of the data file
    this_data_file.seekg(0, std::ios::end);
    this_data_file_len = (uint64_t)this_data_file.tellg();
    this_data_file.seekg(0, std::ios::beg);

    // construct the meta file path, which needs to be at the same relative path as the data file
    filename_no_ext = file_path.stem().string();
    meta_file_name  = filename_no_ext + ".sigmf-meta";
    meta_file_path  = file_path.replace_filename(meta_file_name).string();

    // verify the meta file path and open the meta file
    ok = this->path_is_file(meta_file_path);
    if ( ! ok ) { goto FAILED_CLOSE_DATA_FILE; }

    this_meta_file.open(meta_file_path, std::ios::in);
    if ( ! this_meta_file.is_open() ) { goto FAILED_CLOSE_DATA_FILE; }

    {
        // load the contents of the meta file and parse it into a json document
        std::stringstream meta_json = std::stringstream();
        meta_json << this_meta_file.rdbuf();

        this_meta_file.close();

        this_meta_json.Parse(meta_json.str().c_str());
        if ( this_meta_json.HasParseError() ) { goto FAILED_CLOSE_DATA_FILE; }

        rapidjson::Value meta_global   = rapidjson::Value();
        rapidjson::Value meta_captures = rapidjson::Value();
        rapidjson::Value meta_signals  = rapidjson::Value();
        std::string      meta_dtype    = std::string("");

        // verify that required top-level json keys exists
        if ( ! this_meta_json.HasMember("global"     ) ) { goto FAILED_CLOSE_DATA_FILE; }
        if ( ! this_meta_json.HasMember("captures"   ) ) { goto FAILED_CLOSE_DATA_FILE; }
        if ( ! this_meta_json.HasMember("annotations") ) { goto FAILED_CLOSE_DATA_FILE; }

        meta_global   = this_meta_json["global"];
        meta_captures = this_meta_json["captures"];
        meta_signals  = this_meta_json["annotations"];

        // get the data type and check if it is supported
        if ( ! meta_global.HasMember("core:datatype") ) { goto FAILED_CLOSE_DATA_FILE; }

        meta_dtype = meta_global["core:datatype"].GetString();
        if ( meta_dtype.compare("ci16_le") != 0 ) { goto FAILED; }

        // get the sample rate and center frequency
        if ( ! meta_global.HasMember("core:sample_rate") ) { goto FAILED_CLOSE_DATA_FILE; }
        this_sample_rate = meta_global["core:sample_rate"].GetUint64();

        // get the center frequency within the first capture
        for ( rapidjson::Value::ConstValueIterator itr = meta_captures.Begin(); itr != meta_captures.End(); ++itr )
        {
            const rapidjson::Value& meta_capture = *itr;
            if ( ! meta_capture.HasMember("core:frequency") ) { goto FAILED_CLOSE_DATA_FILE; }

            this_center_freq = meta_capture["core:frequency"].GetUint64();
            break;
        }

        // round up meta-signals
        for ( rapidjson::Value::ConstValueIterator itr = meta_signals.Begin(); itr != meta_signals.End(); ++itr )
        {
            const rapidjson::Value& meta_signal = *itr;

            if ( ! meta_signal.HasMember("core:sample_start"   ) ) { continue; }
            if ( ! meta_signal.HasMember("core:sample_count"   ) ) { continue; }
            if ( ! meta_signal.HasMember("core:freq_lower_edge") ) { continue; }
            if ( ! meta_signal.HasMember("core:freq_upper_edge") ) { continue; }

            sigmf_file::signal signal = sigmf_file::signal();

            signal.sample_start = (uint64_t)meta_signal["core:sample_start"].GetUint64();
            signal.sample_cnt   = (uint64_t)meta_signal["core:sample_count"].GetUint64();
            signal.freq_lower   = (uint64_t)meta_signal["core:freq_lower_edge"].GetDouble();
            signal.freq_upper   = (uint64_t)meta_signal["core:freq_upper_edge"].GetDouble();

            this_signals.push_back(signal);
        }

        // if no meta-signals
        if ( this_signals.empty() ) { goto FAILED_CLOSE_DATA_FILE; }
    }

    this_is_loaded = true;

    return ( true );

FAILED_CLOSE_DATA_FILE:
    this_data_file.close();

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// get sigmf file sample rate and center frequency
uint64_t sigmf_file::get_sample_rate( void ) { return ( this_sample_rate ); }
uint64_t sigmf_file::get_center_freq( void ) { return ( this_center_freq ); }

//--------------------------------------------------------------------------------------------------------------------------

// get sigmf file signals
void sigmf_file::get_signals( std::vector<sigmf_file::signal> &signals )
{
    signals.clear();
    for ( uint32_t i = 0; i < this_signals.size(); i++ ) { signals.push_back(this_signals.at(i)); }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// get samples, at the requested signal offset, from loaded sigmf file
uint8_t* sigmf_file::get_samples_at_signal( sigmf_file::signal signal, uint32_t fft_size, uint32_t *p_samples_len )
{
    uint8_t *p_samples = nullptr;

    if ( ! this_is_loaded ) { goto FAILED; }

    {
        // make sure we are at the start of the file
        this_data_file.seekg(0, std::ios::beg);

        uint64_t            sizeof_sample        = (uint32_t)(sizeof(short) * 2);
        uint64_t            samples_len          = (fft_size * fft_size);
        uint64_t            samples_len_bytes    = (samples_len * sizeof_sample);
        std::complex<float> samples[samples_len] = {};
        uint64_t            samples_offset       = (uint64_t)(samples_len + (signal.sample_start * sizeof_sample));

        // if requested samples would go past the end of the data file
        if ( (samples_offset + samples_len_bytes) > this_data_file_len ) { goto FAILED; }

        // within the data file, move to the sample start offset found within the meta-signal
        this_data_file.seekg(samples_offset, std::ios::beg);

        // read samples from the data file, and convert them from short to float
        for ( uint32_t i = 0; i < samples_len; i++ )
        {
            int16_t int16_real = 0;
            int16_t int16_imag = 0;

            this_data_file.read((char *)&int16_real, sizeof(int16_real));
            if ( this_data_file.fail() ) { goto FAILED; }

            this_data_file.read((char *)&int16_imag, sizeof(int16_imag));
            if ( this_data_file.fail() ) { goto FAILED; }

            float real = (float)(((float)int16_real) / 32768.0f);
            float imag = (float)(((float)int16_imag) / 32768.0f);

            samples[i].real(real);
            samples[i].imag(imag);
        }   

        // return samples in an allocated buffer
        p_samples = (uint8_t *)malloc(sizeof(samples));
        if ( p_samples == nullptr ) { goto FAILED; }

        std::memcpy(p_samples, samples, sizeof(samples));
       *p_samples_len = sizeof(samples);
    }

    return ( p_samples );

FAILED:
   *p_samples_len = 0;

    return ( nullptr );
}

//--------------------------------------------------------------------------------------------------------------------------

// get samples from loaded sigmf file
bool sigmf_file::get_samples( uint8_t *p_buffer, uint32_t buffer_len )
{
    if ( ! this_is_loaded ) { goto FAILED; }

    {
        uint32_t sizeof_sample = (uint32_t)sizeof(std::complex<float>);
        uint32_t samples_len   = (buffer_len / sizeof_sample);
        float   *p_samples     = (float *)p_buffer;

        // if requested samples would go past the end of the data file
        if ( ((uint64_t)this_data_file.tellg() + (buffer_len >> 1)) > this_data_file_len ) { goto FAILED; }

        // read samples from the data file, and convert them from short to float
        for ( uint32_t i = 0; i < samples_len; i++ )
        {
            int16_t int16_real = 0;
            int16_t int16_imag = 0;

            this_data_file.read((char *)&int16_real, sizeof(int16_real));
            if ( this_data_file.fail() ) { goto FAILED; }

            this_data_file.read((char *)&int16_imag, sizeof(int16_imag));
            if ( this_data_file.fail() ) { goto FAILED; }

            float real = (float)(((float)int16_real) / 32768.0f);
            float imag = (float)(((float)int16_imag) / 32768.0f);

           *p_samples++ = real;
           *p_samples++ = imag;
        }   
    }

    return ( true );

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
