//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "gl_multiplot.h"

//-- note ------------------------------------------------------------------------------------------------------------------

// The following projects were leveraged to help produce this source code
//
//   https://gist.github.com/insaneyilin/038a022f2ece61c923315306ddcea081
//   https://medium.com/@csv610/rendering-opencv-frames-using-opengl-textures-1a68f0eb194b

//-- constructor(s) --------------------------------------------------------------------------------------------------------

gl_multiplot::gl_multiplot( bool boxes_plot_on, bool history_plot_on, uint32_t image_width, uint32_t image_height )
{
    // initialize
    this_boxes_plot_on     = boxes_plot_on;
    this_history_plot_on   = history_plot_on;
    this_image_w           = image_width;
    this_image_h           = image_height;
    this_image_pad_w       = 0;
    this_image_pad_h       = 0;
    this_plot_pad_w        = 0;
    this_history_pad_w     = 10;
    this_scaled_image_w    = 0;
    this_scaled_image_h    = 0;
    this_image_diff_col    = 0;
    this_monitor_w         = 0;
    this_monitor_h         = 0;
    this_det_win_w         = 0;
    this_det_win_h         = 0;
    this_ts_win_w          = 0;
    this_ts_win_h          = 0;
    this_ts_max_value      = -1.0f;
    this_ts_max_points     = 256;
    this_ts_x_spacing      = (float)(2.0f / (float)this_ts_max_points);
    this_batch_size        = 0;
    this_batch_rows        = 0;
    this_batch_cols        = 0;
    this_renderer_name     = std::string("");
    this_flush_history     = false;
    this_texture           = 0;
    this_p_det_history_win = nullptr;
    this_p_timeseries_win  = nullptr;
    this_created           = false;

    this_history.clear();
    this_colors.clear();
    this_timeseries_x.clear();
    this_timeseries_y.clear();

    // set colors
    gl_multiplot::color color = gl_multiplot::color();

    color = { .red = 1.0f, .green = 1.0f, .blue = 0.0f, .name = "yellow"       };
    this_colors.push_back(color);
    color = { .red = 1.0f, .green = 0.0f, .blue = 0.0f, .name = "red"          };
    this_colors.push_back(color);
    color = { .red = 0.0f, .green = 0.0f, .blue = 1.0f, .name = "blue"         };
    this_colors.push_back(color);
    color = { .red = 0.8f, .green = 0.4f, .blue = 0.0f, .name = "burnt orange" };
    this_colors.push_back(color);
    color = { .red = 0.6f, .green = 1.0f, .blue = 0.6f, .name = "mint green"   };
    this_colors.push_back(color);
    color = { .red = 0.6f, .green = 0.0f, .blue = 0.0f, .name = "dark red"     };
    this_colors.push_back(color);
    color = { .red = 0.0f, .green = 1.0f, .blue = 1.0f, .name = "cyan"         };
    this_colors.push_back(color);
    color = { .red = 0.7f, .green = 0.4f, .blue = 1.0f, .name = "light purple" };
    this_colors.push_back(color);
    color = { .red = 1.0f, .green = 0.5f, .blue = 0.0f, .name = "orange"       };
    this_colors.push_back(color);
    color = { .red = 0.0f, .green = 0.6f, .blue = 0.0f, .name = "dark green"   };
    this_colors.push_back(color);
    color = { .red = 1.0f, .green = 0.0f, .blue = 1.0f, .name = "hot pink"     };
    this_colors.push_back(color);
    color = { .red = 0.5f, .green = 0.0f, .blue = 1.0f, .name = "dark purple"  };
    this_colors.push_back(color);

    return;    
}

//-- destructor ------------------------------------------------------------------------------------------------------------

gl_multiplot::~gl_multiplot( void )
{
    // clean up
    this->destroy();

    return;
}

//-- public methods --------------------------------------------------------------------------------------------------------

// create the multi-plot
bool gl_multiplot::create( void )
{
    bool ok = false;

    if ( ! this_created )
    {
        // currently support 1024_x_1024 and 2048_x_2048
        if (  this_image_w != this_image_h                    ) { goto FAILED; }
        if ( (this_image_w != 1024) && (this_image_w != 2048) ) { goto FAILED; }

        if ( this_image_w == 1024 )
        {
            this_scaled_image_w = 384;
            this_scaled_image_h = 384;
        }
        else // ( this_image_w == 2048 )
        {
            this_scaled_image_w = 768;
            this_scaled_image_h = 768;
        }

        if ( ! glfwInit() ) { goto FAILED; }

        // get monitor size
        const GLFWvidmode *p_monitor = glfwGetVideoMode(glfwGetPrimaryMonitor());
        if ( p_monitor != nullptr )
        {
            this_monitor_w = (uint32_t)p_monitor->width;
            this_monitor_h = (uint32_t)p_monitor->height;
        }
        else { goto FAILED_GLFW_TERMINATE; }

        // set window dimensions and batch size based on monitor size
        this_batch_cols = (uint32_t)(this_monitor_w / this_scaled_image_w);
        this_batch_rows = 2;

        if ( this_batch_cols & 1 ) { this_batch_cols -= 1; }
        if ( this_batch_cols > 8 ) { this_batch_cols  = 8; }

        if ( (this_boxes_plot_on) && (this_history_plot_on) )
        {
            this_image_pad_h = 10;
            this_det_win_w   = (this_scaled_image_w * this_batch_cols);
            this_det_win_w  += ((this_batch_cols + 1) * this_image_pad_w);
            this_det_win_h   = (this_scaled_image_h * this_batch_rows);
            this_det_win_h  += ((this_batch_rows + 1) * this_image_pad_h);
            this_plot_pad_w  = ((this_history_pad_w * ((this_batch_cols >> 1) - 1)) >> 1);
            this_det_win_w  += (this_plot_pad_w << 1);
            this_ts_win_w    = (this_ts_max_points << 1);
            this_ts_win_h    = this_scaled_image_h;
            this_batch_size  = this_batch_cols;
        }
        else if ( this_boxes_plot_on )
        {
            this_det_win_w  = (this_scaled_image_w * this_batch_cols);
            this_det_win_w += ((this_batch_cols + 1) * this_image_pad_w);
            this_det_win_h  = this_scaled_image_h;
            this_det_win_h  = (this_scaled_image_h * this_batch_rows);
            this_det_win_h += ((this_batch_rows + 1) * this_image_pad_h);
            this_batch_size = (this_batch_cols << 1);
        }
        else if ( this_history_plot_on )
        {
            this_det_win_w  = (this_scaled_image_w * this_batch_cols);
            this_det_win_w += ((this_batch_cols + 1) * this_image_pad_w);
            this_det_win_h  = this_scaled_image_h;
            this_plot_pad_w = ((this_history_pad_w * ((this_batch_cols >> 1) - 1)) >> 1);
            this_det_win_w += (this_plot_pad_w << 1);
            this_ts_win_w   = (this_ts_max_points << 1);
            this_ts_win_h   = this_scaled_image_h;
            this_batch_size = this_batch_cols;
        }

        // prefer windows not be resizable
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        // create and initialize the detections window/plot
        ok = this->init_detections_plot();
        if ( ! ok ) { goto FAILED_GLFW_TERMINATE; }

        if ( this_history_plot_on )
        {
            // create and initialize the time-series window/plot
            ok = this->init_timeseries_plot();
            if ( ! ok ) { goto FAILED_GLFW_TERMINATE; }

            // generate time-series x values
            for ( uint32_t i = 0; i < this_ts_max_points; i++ )
            {
                float timeseries_x = (float)(((float)i * this_ts_x_spacing) - 1.0f);
                this_timeseries_x.push_back(timeseries_x);
            }
        }

        this_created = true;
    }

    return ( true );

FAILED_GLFW_TERMINATE:
    glfwTerminate();

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// destroy the multi-plot
void gl_multiplot::destroy( void )
{
    // clean up
    if ( this_created )
    {
        glDeleteTextures(1, &this_texture);

        if ( this_p_det_history_win != nullptr )
        {
            glfwMakeContextCurrent(this_p_det_history_win);
            glfwDestroyWindow(this_p_det_history_win);
        }

        if ( this_p_timeseries_win != nullptr )
        {
            glfwMakeContextCurrent(this_p_timeseries_win);
            glfwDestroyWindow(this_p_timeseries_win);
        }

        glfwTerminate();
        this_created = false;
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// getters
std::string gl_multiplot::get_renderer_name( void ) { return ( this_renderer_name ); }
uint32_t    gl_multiplot::get_batch_size( void )    { return ( this_batch_size    ); }

//--------------------------------------------------------------------------------------------------------------------------

// udpate all plots
void gl_multiplot::update( std::vector<cv::Mat> &images, std::vector<float> &image_diffs )
{
    if ( (this_created) && (images.size() == this_batch_size) )
    {
        if ( this_flush_history ) // flush history requested ??
        {
            uint32_t history_len = (uint32_t)this_history.size();
            for ( uint32_t i = 0; i < history_len; i++ )
            {
                this_history.at(i).image_1.release();
                this_history.at(i).image_2.release();

                this_colors.push_back(this_history.at(i).color);
            }

            this_history.clear();
            this_timeseries_y.clear();

            this_ts_max_value  = -1.0f;
            this_flush_history = false;
        }
        else
        {
            glfwMakeContextCurrent(this_p_det_history_win);

            // render into the top-level texture in the detections/history window
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, this_texture);

            if ( this_boxes_plot_on   ) { this->update_image_batch(images);          }
            if ( this_history_plot_on ) { this->update_history(images, image_diffs); }

            // render the top-level texture in the detections/history window
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // black
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glMatrixMode(GL_MODELVIEW);

            glBegin(GL_QUADS);

            glTexCoord2i(0, 0);
            glVertex2i(0, 0);
            glTexCoord2i(0, 1);
            glVertex2i(0, this_scaled_image_h);
            glTexCoord2i(1, 1);
            glVertex2i(this_scaled_image_w, this_scaled_image_h);
            glTexCoord2i(1, 0);
            glVertex2i(this_scaled_image_w, 0);

            glEnd();
            glBindTexture(GL_TEXTURE_2D, 0);
            glDisable(GL_TEXTURE_2D);

            glfwSwapBuffers(this_p_det_history_win);

            // render the time-series in the time-series window
            if ( this_history_plot_on )
            {
                glfwMakeContextCurrent(this_p_timeseries_win);
                this->update_timeseries();
                glfwSwapBuffers(this_p_timeseries_win);
            }
        }
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// flush history request
void gl_multiplot::flush_history( void )
{
    this_flush_history = true;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// check for multiplot exit
bool gl_multiplot::check_exit( void )
{
    bool do_exit = false;

    if ( this_created )
    {
        do_exit = glfwWindowShouldClose(this_p_det_history_win);
        if ( (! do_exit) && (this_history_plot_on) )
        {
            do_exit = glfwWindowShouldClose(this_p_timeseries_win);
        }

        glfwPollEvents();
    }

    return ( do_exit );
}

//-- private [static] methods ----------------------------------------------------------------------------------------------

// sort the history vector (descending order based on image difference)
bool gl_multiplot::sort_history( gl_multiplot::history_entry &entry_1, gl_multiplot::history_entry &entry_2 )
{
    return ( entry_1.image_diff > entry_2.image_diff );
}

//-- private methods -------------------------------------------------------------------------------------------------------

// create and initialize the detections/history plot
bool gl_multiplot::init_detections_plot( void )
{
    std::string gl_vendor   = std::string("");
    std::string gl_renderer = std::string("");
    std::string win_title   = std::string("");
    int32_t     frame_w     = 0;
    int32_t     frame_h     = 0;

    // create the detections/history window
    if      ( this_boxes_plot_on && this_history_plot_on ) { win_title = std::string("DETECTIONS / HISTORY"); }
    else if ( this_boxes_plot_on                         ) { win_title = std::string("DETECTIONS");           }
    else if ( this_history_plot_on                       ) { win_title = std::string("HISTORY");              }

    this_p_det_history_win = glfwCreateWindow(this_det_win_w, this_det_win_h, win_title.c_str(), nullptr, nullptr);
    if ( this_p_det_history_win == nullptr ) { goto FAILED; }

    glfwMakeContextCurrent(this_p_det_history_win);
    glfwSwapInterval(1);

    // get vendor and renderer information
    gl_vendor          = std::string((char *)glGetString(GL_VENDOR));
    gl_renderer        = std::string((char *)glGetString(GL_RENDERER));
    this_renderer_name = gl_vendor + " " + gl_renderer;

    // opengl setup for the detections/history window
    glfwMakeContextCurrent(this_p_det_history_win);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, this_scaled_image_w, this_scaled_image_h, 0.0, 0.0, 1.0);
    glMatrixMode(GL_MODELVIEW);

    glfwGetFramebufferSize(this_p_det_history_win, &frame_w, &frame_h);
    glViewport(0, 0, frame_w, frame_h);

    // create the top-level texture for the detection window...sub-image textures are rendered into this texture
    glGenTextures(1, &this_texture);
    glBindTexture(GL_TEXTURE_2D, this_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D(GL_TEXTURE_2D, 0, 3, this_det_win_w, this_det_win_h, 0, GL_BGR, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    return ( true );

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// create and initialize the time-series plot
bool gl_multiplot::init_timeseries_plot( void )
{
    // create the time series (image difference) window
    this_p_timeseries_win = glfwCreateWindow(this_ts_win_w, this_ts_win_h, "IMAGE DIFF TIME SERIES", nullptr, nullptr);
    if ( this_p_timeseries_win == nullptr ) { goto FAILED; }

    glfwMakeContextCurrent(this_p_timeseries_win);
    glfwSwapInterval(1);

    // opengl setup for the time-series window
    glfwMakeContextCurrent(this_p_timeseries_win);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);

    return ( true );

FAILED:
    return ( false );
}

//--------------------------------------------------------------------------------------------------------------------------

// update the batch of images portion of the detections/history plot
void gl_multiplot::update_image_batch( std::vector<cv::Mat> &images )
{
    // render the batch of images into the top-level texture
    if ( this_history_plot_on )
    {
        // when the boxes plot and history plot are both on, the image batch only consumes the first row 
        // of the detections/history window

        uint32_t row = 0;

        for ( uint32_t col = 0; col < this_batch_cols; col++ )
        {
            cv::Mat  scaled_image      = cv::Mat();
            cv::Size scaled_image_size = cv::Size(this_scaled_image_w, this_scaled_image_h);

            cv::resize(images.at(col), scaled_image, scaled_image_size);

            int32_t x = (int32_t)((col * this_scaled_image_w) + ((col + 1) * this_image_pad_w) + this_plot_pad_w);
            int32_t y = (int32_t)((row * this_scaled_image_h) + ((row + 1) * this_image_pad_h));

            glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, this_scaled_image_w, this_scaled_image_h, GL_BGR,
                            GL_UNSIGNED_BYTE, scaled_image.data);
        }
    }
    else // boxes plot only
    {
        // when only the boxes plot is on, the image batch consumes both rows of the detections/history window

        uint32_t image_idx = 0;

        for ( uint32_t row = 0; row < this_batch_rows; row++ )
        {
            for ( uint32_t col = 0; col < this_batch_cols; col++ )
            {
                cv::Mat  scaled_image      = cv::Mat();
                cv::Size scaled_image_size = cv::Size(this_scaled_image_w, this_scaled_image_h);

                cv::resize(images.at(image_idx++), scaled_image, scaled_image_size);

                int32_t x = (int32_t)((col * this_scaled_image_w) + ((col + 1) * this_image_pad_w) + this_plot_pad_w);
                int32_t y = (int32_t)((row * this_scaled_image_h) + ((row + 1) * this_image_pad_h));

                glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, this_scaled_image_w, this_scaled_image_h, GL_BGR,
                                GL_UNSIGNED_BYTE, scaled_image.data);
            }
        }
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// update the history portion of the detections/history plot
void gl_multiplot::update_history( std::vector<cv::Mat> &images, std::vector<float> &image_diffs )
{
    float    image_diff_min  = 0.0f;
    uint32_t image_diffs_len = (uint32_t)image_diffs.size();
    uint32_t history_len     = (uint32_t)this_history.size();

    // find all image pairs that differ by more than the current history minimum, and add them to the history
    if ( history_len > 0 ) { image_diff_min = this_history.at(history_len - 1).image_diff; }

    for ( uint32_t i = 0; i < image_diffs_len; i++ )
    {
        float ts_value = (image_diffs.at(i) / 15.0f);
        float ts_y     = (float)((ts_value - 2.0f) + ts_value);

        if ( ts_y > this_ts_max_value ) { this_ts_max_value = ts_y; }
        gl_multiplot::image_diff image_diff = { .tagged = false, .color = gl_multiplot::color(), .value = ts_y };

        if ( image_diffs.at(i) > image_diff_min )
        {
            image_diff.tagged = true;
            image_diff.color  = this_colors.at(0);

            this_colors.erase(this_colors.begin());
            gl_multiplot::history_entry entry = gl_multiplot::history_entry();

            entry.tagged     = image_diff.tagged;
            entry.color      = image_diff.color;
            entry.image_1    = images.at(i).clone();     // deep copy
            entry.image_2    = images.at(i + 1).clone(); // deep copy
            entry.image_diff = image_diffs.at(i);

            // by color, associate this history entry with the (image diff) time-series
            cv::Point left_point(0, 3);
            cv::Point right_point(this_image_w, 3);

            double red   = (double)(image_diff.color.red   * 255.0f);
            double green = (double)(image_diff.color.green * 255.0f);
            double blue  = (double)(image_diff.color.blue  * 255.0f);

            cv::Scalar color(blue, green, red);

            cv::line(entry.image_1, left_point, right_point, color, /*line width*/7.0f);
            cv::line(entry.image_2, left_point, right_point, color, /*line width*/7.0f);

            this_history.push_back(entry);
        }

        this_timeseries_y.push_back(image_diff);
    }

    // sort the history, in descending order, based on image pair difference
    std::sort(this_history.begin(), this_history.end(), sort_history);

    // if there is more history than can be rendered, trim the list and free resources
    history_len = (uint32_t)this_history.size();
    if ( history_len > (this_batch_size >> 1) )
    {
        for ( uint32_t i = (this_batch_size >> 1); i < history_len; i++ )
        {
            this_history.at(i).image_1.release();
            this_history.at(i).image_2.release();

            this_colors.push_back(this_history.at(i).color);
        }

        this_history.erase((this_history.begin() + (this_batch_size >> 1)), this_history.end());
    }

    // render current history into the top-level texture
    uint32_t row =  0;
    int32_t  col = -1;

    if ( this_boxes_plot_on ) { row++; }

    history_len = (uint32_t)this_history.size();
    for ( uint32_t i = 0; i < history_len; i++ )
    {
        cv::Mat  scaled_image_1    = cv::Mat();
        cv::Mat  scaled_image_2    = cv::Mat();
        cv::Size scaled_image_size = cv::Size(this_scaled_image_w, this_scaled_image_h);

        cv::resize(this_history.at(i).image_1, scaled_image_1, scaled_image_size);
        cv::resize(this_history.at(i).image_2, scaled_image_2, scaled_image_size);

        col++;
        glTexSubImage2D(GL_TEXTURE_2D, 0, ((col * this_scaled_image_w) + ((col >> 1) * this_history_pad_w)),
                        ((row * this_scaled_image_h) + ((row + 1) * this_image_pad_h)), this_scaled_image_w,
                        this_scaled_image_h, GL_BGR, GL_UNSIGNED_BYTE, scaled_image_1.data);
        col++;
        glTexSubImage2D(GL_TEXTURE_2D, 0, ((col * this_scaled_image_w) + ((col >> 1) * this_history_pad_w)),
                        ((row * this_scaled_image_h) + ((row + 1) * this_image_pad_h)), this_scaled_image_w,
                        this_scaled_image_h, GL_BGR, GL_UNSIGNED_BYTE, scaled_image_2.data);
    }

#if 0 // original algo...find image pair with greatest difference and render in next "slot"

    float    image_diff_max     = 0.0f;
    uint32_t image_diff_max_idx = 0;
    uint32_t num_image_diffs    = image_diffs.size();

    // find the image pair with the greatest difference
    for ( uint32_t i = 0; i < num_image_diffs; i++ )
    {
        if ( image_diffs.at(i) > image_diff_max )
        {
            image_diff_max     = image_diffs.at(i);
            image_diff_max_idx = i;
        }
    }

    // render the image pair, with the greatest difference, into the top-level texture
    uint32_t row = 1;

    for ( uint32_t col = this_image_diff_col; col < (this_image_diff_col + 2); col++ )
    {
        cv::Mat  scaled_image      = cv::Mat();
        cv::Size scaled_image_size = cv::Size(this_scaled_image_w, this_scaled_image_h);

        cv::resize(images.at(image_diff_max_idx++), scaled_image, scaled_image_size);

        glTexSubImage2D(GL_TEXTURE_2D, 0, ((col * this_scaled_image_w) + ((col >> 1) * this_history_pad_w)),
                        ((row * this_scaled_image_h) + ((row + 1) * this_image_pad_h)), this_scaled_image_w,
                        this_scaled_image_h, GL_BGR, GL_UNSIGNED_BYTE, scaled_image.data);
    }

    this_image_diff_col += 2;
    if ( this_image_diff_col >= this_batch_cols ) { this_image_diff_col = 0; }

#endif

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

// update the time-series plot
void gl_multiplot::update_timeseries( void )
{
    uint32_t ts_len = (uint32_t)this_timeseries_y.size();

    // trim the oldest points
    if ( ts_len > this_ts_max_points )
    {
        this_timeseries_y.erase(this_timeseries_y.begin(), (this_timeseries_y.begin() + (ts_len - this_ts_max_points)));
    }
    ts_len = (uint32_t)this_timeseries_y.size();

    // render the time-series
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // black
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glColor3f(1.0f, 1.0f, 1.0f); // white
    glLineWidth(1.0f);

    glBegin(GL_LINE_STRIP);
    for ( uint32_t i = 0; i < ts_len; i++ )
    {
        glVertex2f(this_timeseries_x.at(i), this_timeseries_y.at(i).value);
    }
    glEnd();

    // render a line at the current maximum
    glColor3f(1.0f, 1.0f, 1.0f); // white
    glLineWidth(2.0f);

    glBegin(GL_LINES);
    glVertex2f(-1.0f, this_ts_max_value);
    glVertex2f( 1.0f, this_ts_max_value);
    glEnd();

    // render a smale triangle above each time-series value that is tagged
    glBegin(GL_TRIANGLES);
    for ( uint32_t i = 0; i < ts_len; i++ )
    {
        if ( this_timeseries_y.at(i).tagged )
        {
            float red   = this_timeseries_y.at(i).color.red;
            float green = this_timeseries_y.at(i).color.green;
            float blue  = this_timeseries_y.at(i).color.blue;

            glColor3f(red, green, blue);

            float timeseries_x = this_timeseries_x.at(i);
            float timeseries_y = this_timeseries_y.at(i).value;

            glVertex2f(timeseries_x, (timeseries_y + 0.005));
            glVertex2f((timeseries_x - 0.025f), (timeseries_y + 0.07f));
            glVertex2f((timeseries_x + 0.025f), (timeseries_y + 0.07f));
        }
    }
    glEnd();

    return;
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
