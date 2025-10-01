cmake_minimum_required( VERSION 3.12 )
project( tensorrt_yolo )

#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------

# building for nvidia jetson platform ??
if ( ${NVIDIA_JETSON} )
    add_definitions( -DNVIDIA_JETSON )
endif()

# set additional C++ build flags
set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -ffast-math -Wno-deprecated-declarations" )

# set additional CUDA build flags
set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --objdir-as-tempdir --optimize 3 -diag-suppress 611" )

if ( NVIDIA_JETSON )

    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_87,code=sm_87" ) # Jetson AGX ORIN, NX, Nano
    set( BUILD_CUDA_ARCHITECTURES "87" )

else()

##    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70"   ) # Volta V100
##    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75"   ) # Turing T4
    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80"   ) # Ampere A30, A100
    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86"   ) # Ampere A2, A10, A16, A40
    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_89,code=sm_89"   ) # Lovelace L4, L40
    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_90,code=sm_90"   ) # Hopper H100, H200, GH200
    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_100,code=sm_100" ) # Blackwell B200, GB200

##    set( BUILD_CUDA_ARCHITECTURES "70;75;80;86;89;90" )
    set( BUILD_CUDA_ARCHITECTURES "70;75;80;86;89;90;100" )

endif()

# verbose logging enabled ??
if ( ${WITH_VERBOSE_LOGGING} )
    add_definitions( -DVERBOSE_LOGGING )
endif()

# find required packages
find_package( CUDA   REQUIRED )
find_package( OpenCV REQUIRED )
find_package( OpenGL REQUIRED )
find_package( GLUT   REQUIRED )
find_package( glfw3  REQUIRED )

find_library( TENSORRT_NVINFER NAMES nvinfer REQUIRED )
if ( TENSORRT_NVINFER )
    message( "-- Found TENSORRT-NVINFER => ${TENSORRT_NVINFER}" )
else()
    message( FATAL_ERROR "TENSORRT-NVINFER NOT FOUND !!" )
endif()

find_library( MONGODB_BSON NAMES bsoncxx REQUIRED )
if ( MONGODB_BSON )
    message( "-- Found MONGODB-BSON => ${MONGODB_BSON}" )
    set( MONGODB_BSON_INCLUDE_DIRS /usr/local/include/bsoncxx/v_noabi )
else()
    message( FATAL_ERROR "MONGODB-BSON NOT FOUND !!" )
endif()

find_library( MONGODB_CLIENT NAMES mongocxx REQUIRED )
if ( MONGODB_CLIENT )
    message( "-- Found MONGODB-CLIENT => ${MONGODB_CLIENT}" )
    set( MONGODB_CLIENT_INCLUDE_DIRS /usr/local/include/mongocxx/v_noabi )
else()
    message( FATAL_ERROR "MONGODB-CLIENT NOT FOUND !!" )
endif()

if ( ${BUILD_TARGET} STREQUAL "TYE_STREAM_PROCESSOR" )

    find_library( SIGNALHOUND_SM_API NAMES sm_api REQUIRED )
    if ( SIGNALHOUND_SM_API )
        message( "-- Found SIGNALHOUND SM API => ${SIGNALHOUND_SM_API}" )
    else()
        message( FATAL_ERROR "SIGNALHOUND SM API NOT FOUND !!" )
    endif()

endif()

# set include paths
include_directories( ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${OpenGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS}
                     ${glfw3_INCLUDE_DIRS} ${MONGODB_BSON_INCLUDE_DIRS} ${MONGODB_CLIENT_INCLUDE_DIRS}
                     src/common/cuda/spectrogram src/common/cuda/fft_bins_proc src/common/cuda/rssi
                     src/apps/tye_app/tye_lib src/apps/tye_app )

# set output directory for all built targets
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )

#-- tensor-rt yolo engine streaming processor app --------------------------------------------------------------------------

set( TYE_APP_SRC src/common/cuda/spectrogram/cuda_spectrogram.cu
                 src/common/cuda/fft_bins_proc/cuda_fft_bins_proc.cu
                 src/common/cuda/psd_est_proc/cuda_psd_est_proc.cu
                 src/common/cuda/rssi/rssi.cu
                 src/apps/tye_app/tye_lib/cuda_preprocess_image.cu
                 src/apps/tye_app/tye_lib/tye_buffer.cpp
                 src/apps/tye_app/tye_lib/tye_buffer_pool.cpp
                 src/apps/tye_app/tye_lib/tye_spectrogram.cpp
                 src/apps/tye_app/tye_lib/tye_spectrogram_pool.cpp
                 src/apps/tye_app/tye_lib/tye_ops.cpp
                 src/apps/tye_app/tye_lib/tye_notifier.cpp
                 src/apps/tye_app/tye_lib/tye_processor.cpp
                 src/apps/tye_app/tye_lib/tye_cluster.cpp
                 src/apps/tye_app/gl_multiplot.cpp
                 src/apps/tye_app/config.cpp
                 src/apps/tye_app/mongodb_sink.cpp
                 src/apps/tye_app/engine.cpp )

if ( ${BUILD_TARGET} STREQUAL "TYE_STREAM_PROCESSOR" )

    set( TYE_APP_SRC ${TYE_APP_SRC} src/apps/tye_app/tye_sp/sh_sm_radio.cpp
                                    src/apps/tye_app/tye_sp/sp_config.cpp
                                    src/apps/tye_app/tye_sp/sp_frontend.cpp
                                    src/apps/tye_app/tye_sp/tye_sp.cpp )

    add_definitions( -DTYE_STREAM_PROCESSOR )
    set( TYE_APP_NAME "tye_sp" )

elseif ( ${BUILD_TARGET} STREQUAL "TYE_FILE_PROCESSOR" )

    set( TYE_APP_SRC ${TYE_APP_SRC} src/apps/tye_app/tye_fp/file_base.cpp
                                    src/apps/tye_app/tye_fp/sigmf_file.cpp
                                    src/apps/tye_app/tye_fp/fp_config.cpp
                                    src/apps/tye_app/tye_fp/fp_frontend.cpp
                                    src/apps/tye_app/tye_fp/tye_fp.cpp )

    add_definitions( -DTYE_FILE_PROCESSOR )
    set( TYE_APP_NAME "tye_fp" )

endif()

set( CUDA_LIBS     cufft                 )
set( TENSORRT_LIBS nvinfer               )
set( GL_LIBS       GL GLU glfw           )
set( DB_LIBS       bsoncxx mongocxx      )
set( RADIO_LIBS    ${SIGNALHOUND_SM_API} )

set( TYE_APP_LIBS ${CUDA_LIBS} ${TENSORRT_LIBS} ${OpenCV_LIBS} ${GL_LIBS} ${DB_LIBS} ${RADIO_LIBS} )

cuda_add_executable(   ${TYE_APP_NAME} ${TYE_APP_SRC}  )
set_target_properties( ${TYE_APP_NAME} PROPERTIES CUDA_ARCHITECTURES "${BUILD_CUDA_ARCHITECTURES}" )
target_link_libraries( ${TYE_APP_NAME} ${TYE_APP_LIBS} )

#-- tensor-rt yolo engine streaming processor retune via advertisement app -------------------------------------------------

set( TYE_SP_AD_RETUNE_APP_SRC src/apps/tye_sp_ad_retune/tye_sp_ad_retune.cpp )
add_executable( tye_sp_ad_retune ${TYE_SP_AD_RETUNE_APP_SRC} )

#-- tensor-rt yolo engine streaming processor update parameters via advertisement app --------------------------------------

set( TYE_SP_AD_UPDATE_PARAMS_APP_SRC src/apps/tye_sp_ad_update_params/tye_sp_ad_update_params.cpp )
add_executable( tye_sp_ad_update_params ${TYE_SP_AD_UPDATE_PARAMS_APP_SRC} )

#---------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------
