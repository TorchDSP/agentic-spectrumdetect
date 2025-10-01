# TensorRT YOLO Engine (tye)

**Tensor RT YOLO Engine Streaming Processor (tye_sp)** is a C++/CUDA based spectrum analyzer that leverages an int8 quantized version of YOLO 11 Small trained on TorchSig synthetic data to detect signals in either 50Ms or 25Ms streaming IQ from a Signal Hound USB SM series device:

* Comes with an optional OpenGL based GUI for spectral detection (--boxes-plot) and spectral change point detection (--history-plot).
* Includes both YOLO based spectrogram signal detection and average spectrum power change point detection.
* Calculates and reports the RSSI values in the YOLO boxes.
* Reports spectrum results to a Mongo database.
* Talks to Signal Hound SM series USB only at the moment, UHD coming soon. Can be reconfigured on the fly over UDP messages for Agentic or programatic control.


---

## Directions

1. Install the Prerequisites specified in the Docker container. The Docker container will build but not run at the moment, the test devices we have were crashing when we tried to access USB through the container. UHD and network based SM series support is coming soon.  
2. Run this script to download the 11s.pt trained model from TorchSig trained_model_download.sh
3. Run "./model_pt_to_trt_engine_int8 11s.pt 1" to convert the 11s.pt model to an int8 TensorRT engine. Note you need to do this for every GPU you plan to run it on for optimal speed. You need ultralytics==v8.3.120 or earlier, because the quantization algorithm was changed from EntropyCalibtration2 to minmax. Minmax does not work with this code. The script will use Ultralytics to make a quantization cache file and then export the model to ONNX and then TensorRT will load that in. It quantizes based on the COCO dataset. This works because during training on TorchSig, the training was started from the COCO pre-trained weights provided by Ultralytics and the first layer was frozen.
4. Run "./do_run_cmake -t tye_sp -c /usr/local/cuda-12.9/" and cd build and make.
5. Then to start tye_sp run something like: bin/tye_sp --gpus 1 --engine-path 11s.int8.gpu1.engine   --engines-per-gpu 2    --sample-rate-mhz 50 --center-freq 2450000000 --pixel-min-val -100.0 --pixel-max-val -10.0 --boxes-plot --database-off --history-plot --atten-db 10 --ref-level -20
6. This code was tested for Nvidia A10's and assumes at least two GPU's for the Agentic and Tye use. GPU 0 runs GPT-OSS-20B and GPU 1 runs tye_sp. We are using bfloat16 and have tested primarily with CUDA 12.9. An Ampere series Nvidia GPU or newer is required. 
---

## Prerequisites

* CUDA 12.9.1, CUDNN, Ubuntu 24.04 recomended 
* ultralytics==v8.3.120
* opencv and opencv_contrib 4.12.0 compiled for cuda support
* mongo-css-driver 4.1.0
* Signal Hound SDK 09_11_25

---


## License

Agentic Spectrumdetect is released under the MIT License. The MIT license is a popular open-source software license enabling free use, redistribution, and modifications, even for commercial purposes, provided the license is included in all copies or substantial portions of the software. Agentic Spectrumdetect has no connection to MIT, other than through the use of this license.

---

