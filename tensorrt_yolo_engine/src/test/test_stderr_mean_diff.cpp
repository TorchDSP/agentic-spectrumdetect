//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------

#include "tye_app_includes.h"
#include "cuda_fft_bins_proc.h"

//-- constants -------------------------------------------------------------------------------------------------------------

static const uint32_t THIS_FFT_SIZE      = 1024;
static const uint32_t THIS_NUM_FFT_BINS  = (THIS_FFT_SIZE * THIS_FFT_SIZE);
static const uint32_t THIS_NUM_TEST_ITER = 25;

//-- variables -------------------------------------------------------------------------------------------------------------

static std::vector<float> this_fft_bins_1    = std::vector<float>();
static std::vector<float> this_fft_bins_2    = std::vector<float>();
static std::vector<float> this_pow_diff      = std::vector<float>();
static std::vector<float> this_pow_diff_sqrd = std::vector<float>();

//-- support functions -----------------------------------------------------------------------------------------------------

static void generate_fft_bins( void )
{
    std::cout << ">> GENERATING FFT BINS" << std::endl << std::flush;
    srand(time(nullptr));

    float min_float = 0.0f;
    float max_float = 255.0f;

    for ( uint32_t i = 0; i < THIS_NUM_FFT_BINS; i++ )
    {
        float rand_float_1 = (min_float + ((float)rand() / (float)RAND_MAX) * (max_float - min_float));
        float rand_float_2 = (min_float + ((float)rand() / (float)RAND_MAX) * (max_float - min_float));

        this_fft_bins_1.push_back(rand_float_1);
        this_fft_bins_2.push_back(rand_float_2);
    }

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

static void run_test_host( void )
{
    float    mean_diff = 0.0f;
    uint32_t num_iter  = THIS_NUM_TEST_ITER;

    std::cout << ">> RUNNING TEST ON HOST => " << std::flush;
    auto start_time = std::chrono::system_clock::now();

    for ( uint32_t loop_cnt = 0; loop_cnt < num_iter; loop_cnt++ )
    {
        for ( uint32_t i = 0; i < THIS_NUM_FFT_BINS; i++ )
        {
            float pow_diff      = (this_fft_bins_1.at(i) - this_fft_bins_2.at(i));
            float pow_diff_sqrd = (pow_diff * pow_diff);

            this_pow_diff.push_back(pow_diff);
            this_pow_diff_sqrd.push_back(pow_diff_sqrd);
        }

        float pow_diff_sum      = 0.0;
        float pow_diff_sqrd_sum = 0.0;

        for ( uint32_t i = 0; i < this_pow_diff.size();      i++ ) { pow_diff_sum      += this_pow_diff.at(i);      }
        for ( uint32_t i = 0; i < this_pow_diff_sqrd.size(); i++ ) { pow_diff_sqrd_sum += this_pow_diff_sqrd.at(i); }

        mean_diff += sqrtf((THIS_NUM_FFT_BINS * pow_diff_sqrd_sum) - (pow_diff_sum * pow_diff_sum)) / THIS_NUM_FFT_BINS;

        this_pow_diff.clear();
        this_pow_diff_sqrd.clear();
    }

    auto   end_time = std::chrono::system_clock::now();
    double num_usec = (double)std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    std::cout << (mean_diff / num_iter) << " [AVG " << (num_usec / num_iter) << " USEC]" << std::endl << std::flush;

    return;
}

//--------------------------------------------------------------------------------------------------------------------------

static void run_test_gpu( void )
{
    uint32_t     gpu       = 0;
    float        mean_diff = 0.0f;
    uint32_t     num_iter  = THIS_NUM_TEST_ITER;
    cudaStream_t stream    = {};

    std::cout << ">> RUNNING TEST ON GPU  => " << std::flush;
    cudaSetDevice(gpu);

    cudaError_t status = cudaStreamCreate(&stream);
    if ( status != cudaSuccess ) { throw std::runtime_error("[EXCEPTION] CREATING CUDA STREAM"); }

    cuda_fft_bins_proc *p_proc = new cuda_fft_bins_proc(gpu, THIS_NUM_FFT_BINS);
    if ( p_proc == nullptr ) { throw std::runtime_error("[EXCEPTION] CREATING CUDA FFT BINS PROC"); }

    auto start_time = std::chrono::system_clock::now();

    for ( uint32_t loop_cnt = 0; loop_cnt < num_iter; loop_cnt++ )
    {
        mean_diff += p_proc->compute_stderr_mean_diff(this_fft_bins_1.data(), this_fft_bins_2.data(), stream);
    }

    auto   end_time = std::chrono::system_clock::now();
    double num_usec = (double)std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    std::cout << (mean_diff / num_iter) << " [AVG " << (num_usec / num_iter) << " USEC]" << std::endl << std::flush;

    cudaStreamDestroy(stream);

    return;
}

//-- entry point -----------------------------------------------------------------------------------------------------------

int32_t main( int32_t num_args, char *p_args[] )
{
    std::cout << std::endl << std::flush;

    generate_fft_bins();
    run_test_host();
    run_test_gpu();

    std::cout << std::endl << std::flush;

    return ( 0 );
}

//--------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------
