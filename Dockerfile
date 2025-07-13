FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Consolidated environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/app/venv/bin:$PATH"

WORKDIR /app

# Single RUN layer for system dependencies - Ubuntu 22.04 ships with Python 3.10.12
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        ca-certificates \
        curl \
        git \
        pkg-config \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy FFmpeg binaries in single layer
COPY --from=mwader/static-ffmpeg:7.1.1 /ffmpeg /ffprobe /usr/local/bin/
RUN chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe

# Create venv and install Python dependencies
RUN python3.10 -m venv venv && \
    pip install --upgrade pip setuptools wheel

# Copy requirements first for better cache utilization
COPY requirements.txt .
RUN pip install -r requirements.txt

# Uncomment and modify for model preloading during build
# ARG HF_TOKEN
# ENV HF_TOKEN=${HF_TOKEN}
# RUN python -c "
#     from huggingface_hub import snapshot_download
#     snapshot_download(
#         repo_id='deepdml/faster-whisper-large-v3-turbo-ct2',
#         local_dir='models',
#         local_dir_use_symlinks=False,
#         resume_download=True
#     )"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

COPY --chown=appuser:appuser handler.py .

STOPSIGNAL SIGINT
CMD ["python", "-u", "handler.py"]