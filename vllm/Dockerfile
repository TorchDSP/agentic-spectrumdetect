FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update 
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    curl
RUN    pip install --break-system-packages uv
RUN    uv pip install ray[serve] --system --break-system-packages
RUN    git clone -b v0.11.1rc0 https://github.com/vllm-project/vllm.git
RUN    cd vllm && VLLM_USE_PRECOMPILED=1 uv pip install --system --break-system-packages --editable . 
RUN    apt-get clean && rm -rf /var/lib/apt/lists/*

EXPOSE 8888

#ENV VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
ENV VLLM_ATTENTION_BACKEND=TRITON_ATTN
ENV LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH

RUN ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /usr/local/lib/libcuda.so

#CMD ["vllm", "serve", "openai/gpt-oss-20b", "-tp", "1", "--tool-call-parser", "openai", "--reasoning-parser", "openai_gptoss", "--enable-auto-tool-choice", "--host", "\"0.0.0.0\"", "--port", "\"8888\"", "--max-model-len", "131072", "--gpu-memory-utilization", "0.90",   "--served-model-name", "gpt-oss-20b",   "--api-key", "my-key"]

ENTRYPOINT ["vllm"]
