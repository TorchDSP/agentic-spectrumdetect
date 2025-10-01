#ifndef INCLUDE_GL_MULTIPLOT_H
#define INCLUDE_GL_MULTIPLOT_H

//--------------------------------------------------------------------------------------------------------------------------

#include "tye_app_includes.h"

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

class gl_multiplot
{
public: //==================================================================================================================

    // constructor(s) / destructor
    gl_multiplot( bool boxes_plot_on, bool history_plot_on, uint32_t image_width, uint32_t image_height );
   ~gl_multiplot( void );

    // public methods
    bool        create( void );
    void        destroy( void );
    std::string get_renderer_name( void );
    uint32_t    get_batch_size( void );
    void        update( std::vector<cv::Mat> &images, std::vector<float> &image_diffs );
    void        flush_history( void );
    bool        check_exit( void );

private: //=================================================================================================================

    // private constants
    const std::string NAME = std::string("GL_MULTIPLOT");

    // private types
    typedef struct
    {
        float       red;
        float       green;
        float       blue;
        std::string name;

    } color;

    typedef struct
    {
        bool                tagged;
        gl_multiplot::color color;
        float               value;

    } image_diff;

    typedef struct
    {
        bool                tagged;
        gl_multiplot::color color;
        cv::Mat             image_1;
        cv::Mat             image_2;
        float               image_diff;

    } history_entry;

    // private variables
    bool        this_boxes_plot_on;
    bool        this_history_plot_on;
    uint32_t    this_image_w;
    uint32_t    this_image_h;
    uint32_t    this_image_pad_w;
    uint32_t    this_image_pad_h;
    uint32_t    this_plot_pad_w;
    uint32_t    this_history_pad_w;
    uint32_t    this_scaled_image_w;
    uint32_t    this_scaled_image_h;
    uint32_t    this_image_diff_col;
    uint32_t    this_monitor_w;
    uint32_t    this_monitor_h;
    uint32_t    this_det_win_w;
    uint32_t    this_det_win_h;
    uint32_t    this_ts_win_w;
    uint32_t    this_ts_win_h;
    float       this_ts_max_value;
    uint32_t    this_ts_max_points;
    float       this_ts_x_spacing;
    uint32_t    this_batch_size;
    uint32_t    this_batch_rows;
    uint32_t    this_batch_cols;
    std::string this_renderer_name;
    bool        this_flush_history;
    GLuint      this_texture;
    GLFWwindow *this_p_det_history_win;
    GLFWwindow *this_p_timeseries_win;
    bool        this_created;

    std::vector<gl_multiplot::history_entry> this_history;
    std::vector<gl_multiplot::color>         this_colors;
    std::vector<float>                       this_timeseries_x;
    std::vector<gl_multiplot::image_diff>    this_timeseries_y;

    // private [static] methods
    static bool sort_history( gl_multiplot::history_entry &entry_1, gl_multiplot::history_entry &entry_2 );

    // private methods
    bool init_detections_plot( void );
    bool init_timeseries_plot( void );
    bool init_plots( void );
    void update_image_batch( std::vector<cv::Mat> &images );
    void update_history( std::vector<cv::Mat> &images, std::vector<float> &image_diffs );
    void update_timeseries( void );
};

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#endif // INCLUDE_GL_MULTIPLOT_H
