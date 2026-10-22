# Docker image for IGM (TensorFlow backend).
#
# The TensorFlow version of the base image must match the pin in setup.py
# (tensorflow[and-cuda]==2.17.0). The same image runs on CPU and on NVIDIA GPU
# hosts (run with `--gpus all`); TensorFlow ships its CUDA 12 libraries.
#
# Build (pass your uid/gid so files written from the container are yours):
#   docker build --no-cache -t igm --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
# Run:
#   docker run --gpus all -it -v $PWD:/home/igmuser/work -w /home/igmuser/work igm
#
# Notes:
#   - GPUs older than Pascal (e.g. Maxwell / GTX TITAN X) are not supported by
#     TensorFlow >= 2.16 wheels; Blackwell (sm_120) needs an NGC base image
#     (nvcr.io/nvidia/tensorflow:25.02-tf2-py3 or later) instead of this one.
#
FROM tensorflow/tensorflow:2.17.0-gpu

ARG USER_ID=1000
ARG GROUP_ID=1000

ENV TZ=Europe/Paris \
    DEBIAN_FRONTEND=noninteractive \
    # Quieter TensorFlow logs
    TF_CPP_MIN_LOG_LEVEL=3 \
    # Allocate GPU memory on demand instead of grabbing the whole GPU
    # (needed if several runs share one GPU). No effect on CPU-only hosts.
    TF_FORCE_GPU_ALLOW_GROWTH=true

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
 && apt-get update \
 && apt-get install -y --no-install-recommends git nano curl tree \
 && rm -rf /var/lib/apt/lists/*

RUN addgroup --gid $GROUP_ID igmuser \
 && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID igmuser

USER igmuser
WORKDIR /home/igmuser/

RUN pip install --no-cache-dir -U pip \
 && git clone https://github.com/instructed-glacier-model/igm.git \
 && pip install --no-cache-dir -e ./igm

ENV PATH="/home/igmuser/.local/bin:${PATH}"

CMD ["bash"]
