FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set Work Directory
WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=True
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Update, upgrade, install packages and clean up
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    pkg-config \
    zip \
    build-essential \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-distutils && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy static FFmpeg binaries from official image
COPY --from=mwader/static-ffmpeg:7.1.1 /ffmpeg /usr/local/bin/ffmpeg
COPY --from=mwader/static-ffmpeg:7.1.1 /ffprobe /usr/local/bin/ffprobe
RUN chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe

RUN python3.10 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Install requirements in smaller chunks to avoid memory issues
RUN pip install --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip install -r requirements.txt

# ------------------- download model weights during build -------------------
# This places the model inside /app/models so runtime startup is instant.
# If the model is gated, pass a token at build time:  
#   docker build --build-arg HF_TOKEN=xxx -t myimage .
# The python snippet picks up that token automatically via the env-var.
# ARG HF_TOKEN
# ENV HF_TOKEN=${HF_TOKEN}
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='deepdml/faster-whisper-large-v3-turbo-ct2', local_dir='models', local_dir_use_symlinks=False, resume_download=True)"
# ---------------------------------------------------------------------------

COPY handler.py .

STOPSIGNAL SIGINT

CMD ["python", "-u", "handler.py"]
