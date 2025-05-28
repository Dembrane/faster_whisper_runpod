FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    git \
    net-tools \
    wget \
    nano \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy static FFmpeg binaries from official image
COPY --from=mwader/static-ffmpeg:7.1.1 /ffmpeg /usr/local/bin/ffmpeg
COPY --from=mwader/static-ffmpeg:7.1.1 /ffprobe /usr/local/bin/ffprobe
RUN chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PIP_CACHE_DIR=/home/user/.cache/pip

WORKDIR $HOME/app

RUN pip install --upgrade pip

RUN pip install "setuptools>=64.0.0" wheel "setuptools_scm>=8.0" --upgrade
# RUN pip install torch==2.5.1+cu121 torchvision==0.20.1 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# RUN pip install whisperx==3.3.4

COPY --chown=user ./requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY --chown=user . $HOME/app

# for debugging
# CMD ["tail", "-f", "/dev/null"]

RUN pip install -e .

# Run the runpod handler
CMD ["python", "-u", "handler.py"]
