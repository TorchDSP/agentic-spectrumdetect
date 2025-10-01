FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
 
RUN apt update && apt install -y \
    python3-pip \
    tensorrt \
    ffmpeg \
    libsm6 \
    libxext6 \
    cmake \
    vim \
    less \
    git \
    libgl-dev \
    libglx-dev \
    libglfw3-dev \
    freeglut3-dev \
    curl \
    udev \
    wget \
    unzip \
    libssl-dev \
    libqt5x11extras5-dev \
    sudo \
    rapidjson-dev \
    usbutils 

RUN pip install uv --break-system-packages
RUN uv pip install ultralytics==v8.3.120 --system --break-system-packages    
RUN uv pip install onnx --system --break-system-packages    
RUN mkdir /build && cd /build && git clone -b 4.12.0 https://github.com/opencv/opencv.git && git clone -b 4.12.0 https://github.com/opencv/opencv_contrib.git && \
    cd opencv && mkdir build && cd build && \
    export CUDA_COMPUTE_CAPABILITY=8.6 && \
    export PATH_TO_CUDA=/usr/local/cuda-12.9 && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DWITH_CUDA=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON \
        -DWITH_TBB=ON -DCUDA_FAST_MATH=1 -DOPENCV_ENABLE_NONFREE=ON -DENABLE_FAST_MATH=1 -DWITH_CUBLAS=1 \
        -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -DCUDA_ARCH_BIN=$CUDA_COMPUTE_CAPABILITY \
        -DCUDA_TOOLKIT_ROOT_DIR=$PATH_TO_CUDA .. && \
    make -j && \
    make install && \
    ldconfig

RUN cd /build && \
    curl -OL https://github.com/mongodb/mongo-cxx-driver/releases/download/r4.1.0/mongo-cxx-driver-r4.1.0.tar.gz
RUN cd /build && tar -xzf mongo-cxx-driver-r4.1.0.tar.gz
RUN cd /build/mongo-cxx-driver-r4.1.0/build && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 ../
RUN cd /build/mongo-cxx-driver-r4.1.0/build && make -j32 && make install

RUN cd /build && wget https://signalhound.com/sigdownloads/SDK/signal_hound_sdk_09_11_25.zip && wget https://signalhound.com/sigdownloads/Spike/Spike\(Ubuntu22.04x64\)_4_0_8.zip && \
    curl -L -o "11s.pt" "https://bucket.ltsnet.net/torchsig/models/11s.pt"
RUN cd /build && unzip signal_hound_sdk_09_11_25.zip && cd signal_hound_sdk && \
    cp device_apis/sm_series/lib/linux/Ubuntu\ 18.04/libsm_api.so.2.3.8 /usr/local/lib && \
    cp device_apis/sm_series/include/sm_api.h /usr/local/include && \
    cp device_apis/sm_series/lib/linux/sh_usb.rules /etc/udev/rules.d
RUN cd /usr/local/lib && ln -s libsm_api.so.2.3.8 libsm_api.so.2
RUN cd /usr/local/lib && ln -s libsm_api.so.2     libsm_api.so
RUN ldconfig
ADD ./ /build/tensorrt_yolo_engine/
RUN cd /build/tensorrt_yolo_engine && ./do_run_cmake -t tye_sp -c  /usr/local/cuda-12.9  && cd build && make
RUN cd /build && yolo settings datasets_dir='./'
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN rm -rf /build/signal_hound_sdk_09_11_25.zip && rm -rf /build/mongo-cxx-driver-r4.1.0.tar.gz



WORKDIR /build
