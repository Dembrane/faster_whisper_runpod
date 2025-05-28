FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set Work Directory
WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=True
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash



# Update, upgrade, install packages and clean up
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    # Basic Utilities
    bash \
    ca-certificates \
    curl \
    git \
    pkg-config \
    zip \
    build-essential \
    # Python and development tools
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-distutils && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.10 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip and install base dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch separately with extended timeout
RUN pip install --no-cache-dir --timeout=3000 \
    torch==2.0.0 \
    torchvision==0.15.0 \
    torchaudio==2.0.0 \
    -f https://download.pytorch.org/whl/cu118/torch_stable.html

# Install Rust dependencies
RUN pip install --no-cache-dir setuptools-rust==1.8.0

# Copy and install requirements with retry mechanism
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your handler code
COPY handler.py .

# Set Stop signal and CMD
STOPSIGNAL SIGINT
CMD ["python", "-u", "handler.py"]
