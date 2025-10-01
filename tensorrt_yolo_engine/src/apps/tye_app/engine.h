#ifndef INCLUDE_ENGINE_H
#define INCLUDE_ENGINE_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_app_includes.h"
#include "tye_buffer.h"
#include "tye_spectrogram.h"
#include "tye_cluster.h"
#include "gl_multiplot.h"
#include "config.h"
#include "mongodb_sink.h"
#include "cuda_fft_bins_proc.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class engine
{
public: //==================================================================================================================

    // constructor(s) / destructor
    engine( config *p_config, mongodb_sink *p_mongodb_sink, bool test_debug = false );
   ~engine( void );

    // public methods
    bool        start( void (*exit_cb)( void ) );
    void        shutdown( void );
    bool        is_running( void );
    std::string get_gl_renderer_name( void );
    void        process( tye_buffer *p_buffer );

private: //=================================================================================================================

    // private constants
    const std::string NAME = std::string("ENGINE");

    // private types
    typedef struct
    {
        uint64_t                                 id;
        std::string                              source_name;
        uint64_t                                 start_millis;
        uint64_t                                 last_millis;
        std::string                              image_path;
        std::vector<tye_types::detections_tuple> detection_data;

    } detection_group;

    typedef struct
    {
        std::vector<uint64_t>          sequ_nums;
        std::vector<cv::Mat>           images;
        std::vector<tye_spectrogram *> spectrograms;
        std::vector<float>             spectrogram_diffs;

    } detections_batch;

    // private variables
    void (*this_exit_cb)( void );

    std::thread        *this_p_mgr_thread;
    std::thread        *this_p_dproc_thread;
    std::thread        *this_p_ad_thread;
    config             *this_p_config;
    mongodb_sink       *this_p_mongodb_sink;
    gl_multiplot       *this_p_multiplot;
    cuda_fft_bins_proc *this_p_fft_bins_proc;
    cudaStream_t        this_stream;
    int32_t             this_cnc_socket;
    std::string         this_gl_renderer_name;
    bool                this_is_sw_gl_renderer;
    bool                this_test_debug;
    bool                this_process_stream;
    bool                this_database_off;
    uint32_t            this_engines_per_gpu;
    std::string         this_output_path;
    uint32_t            this_engine_cluster_idx;
    uint64_t            this_last_buffer_millis;
    bool                this_buffer_flow_started;
    bool                this_running;
    bool                this_ad_ready;
    bool                this_dproc_ready;
    bool                this_engine_ready;
    bool                this_exit;

    std::mutex                             this_buffer_mutex;
    std::vector<tye_buffer *>              this_buffer_queue;
    std::mutex                             this_detections_in_mutex;
    std::vector<engine::detection_group *> this_detection_groups_in;
    std::mutex                             this_detections_out_mutex;
    std::vector<engine::detection_group *> this_detection_groups_out;
    engine::detections_batch               this_detections_batch;
    std::vector<tye_cluster *>             this_engine_clusters;

    // private [static] methods
    static void notify_detections_cb( tye_types::detections_tuple dt, void *p_data );

    // private methods
    bool open_cnc_socket( void );
    void send_cnc_status( bool success, std::string msg_type, struct sockaddr_in *p_peer_sa_in, ssize_t peer_sa_in_len );
    void handle_cnc_update_params( rapidjson::Document &rj_cnc, struct sockaddr_in *p_peer_sa_in, ssize_t peer_sa_in_len );
    void handle_cnc( void );
    void close_cnc_socket( void );
    void shutdown_engine_clusters( void );
    void handle_detection_groups_done_stream( void );
    void handle_detection_groups_done_file( void );
    void check_file_processing_done( void );
    bool detection_groups_to_process( void );
    void generate_db_detections( tye_types::detections_aux &detections_aux, std::vector<tye_types::detection> &detections,
                                 std::vector<mongodb_sink::detection> &db_detections );
    void process_detection_group_stream( engine::detection_group *p_group );
    void process_detection_group_file( engine::detection_group *p_group );
    void cleanup_detection_groups( void );
    void ad_thread( void );
    void dproc_thread( void );
    void mgr_thread( void );
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_ENGINE_H
