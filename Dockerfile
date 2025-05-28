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

# Create and activate virtual environment
RUN python3.10 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip and install base dependencies
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with smaller memory footprint
RUN pip install --timeout=3000 \
    torch==2.0.0 \
    torchvision==0.15.0 \
    torchaudio==2.0.0 \
    -f https://download.pytorch.org/whl/cu118/torch_stable.html

# Install requirements in smaller chunks to avoid memory issues
COPY requirements.txt .

# Install each package separately to avoid memory issues
RUN pip install whisperx==3.3.4
RUN pip install runpod==1.7.9  
RUN pip install python-dotenv==1.1.0

# Copy your handler code
COPY handler.py .

# Set Stop signal and CMD
STOPSIGNAL SIGINT
CMD ["python", "-u", "handler.py"]
