# RunPod Whisper Transcription Serverless Worker
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/dembrane/runpod-whisper)

This repository contains a serverless worker for RunPod designed to perform audio transcription using WhisperX. It can process audio provided as a base64 encoded string or a direct URL.

## Overview

The `runpod-whisper` worker leverages the power of WhisperX (which uses faster-whisper) to provide fast and accurate audio transcriptions. It's built to be deployed on RunPod as a serverless endpoint, automatically scaling with demand. The worker supports multiple languages, configurable model selection, and can run on CPU or GPU.

## Features

*   **Audio Input Flexibility**: Accepts audio via base64 encoded strings or direct URLs.
*   **WhisperX Powered**: Utilizes WhisperX for efficient and accurate transcription.
*   **Configurable Model**: The specific Whisper model can be chosen via an environment variable (defaults to `Systran/faster-whisper-large-v1`).
*   **Multi-Language Support**: Transcribe audio in various languages. Tokenizers for supported languages are pre-loaded for faster processing.
*   **Automatic Hardware Detection**: Optimally uses available hardware, selecting CUDA for GPU acceleration (with `float16`) or CPU (with `float32` or `int8`).
*   **RunPod Optimized**: Designed as a standard RunPod serverless handler.

## Environment Variables

The behavior of the worker can be configured through the following environment variables:

*   `WHISPER_MODEL_NAME`: The Hugging Face model name for WhisperX to use.
    *   Default: `"Systran/faster-whisper-large-v1"`
*   `TASK`: The task for Whisper (e.g., "transcribe").
    *   Default: `"transcribe"`
*   `DEFAULT_LANGUAGE_CODE`: The default language code to use if none is specified in the request.
    *   Default: `"en"`
*   `SUPPORTED_LANGUAGES`: A comma-separated list of language codes for which tokenizers should be pre-loaded (e.g., "en,nl,es,fr").
    *   Default: `"en,nl"`
*   `DEBUG`: Set to `"true"`, `"1"`, or `"yes"` to enable additional debug logging.
    *   Default: `"false"`

## Handler API

The worker exposes a single handler that expects a JSON payload.

### Input Payload (`event["input"]`)

The `input` object in the JSON payload should contain:

```json
{
    "audio_base_64": "string (optional)",
    "audio": "string (URL, optional)",
    "language": "string (optional, e.g., 'en', 'nl')",
    "initial_prompt": "string (optional)"
}
```

*   `audio_base_64` (string, optional): Base64 encoded audio data.
*   `audio` (string, optional): A URL pointing to an audio file (e.g., MP3).
*   One of `audio_base_64` or `audio` must be provided.
*   `language` (string, optional): The language code of the audio. If not provided, or if the provided language is not in `SUPPORTED_LANGUAGES`, it defaults to `DEFAULT_LANGUAGE_CODE`.
*   `initial_prompt` (string, optional): An initial prompt to guide the transcription model.

**Example Input (`test_input.json`):**
```json
{
	"input": {
		"audio": "https://ams3.digitaloceanspaces.com/dbr-echo-local-uploads/audio-chunks/7a0f0d76-daf5-40c0-8d10-aefa4b335215-e5688e04-fbee-45ef-be15-acdbe38bb6bc-audio_Arthur-%5BAudioTrimmer.com%5D.mp3?AWSAccessKeyId=DO00VRQY3P7N8LCC2CPF&Signature=EfIl7SlxfzaJ%2FE96Nw68x%2B%2FyYnI%3D&Expires=1748509985",
		"language": "nl",
		"initial_prompt": "Hallo, laten we beginnen. Eerst even een introductieronde en dan kunnen we aan de slag met de thema van vandaag."
	}
}
```

### Output Payload

If successful, the handler returns a JSON object:

```json
{
    "model_output": { /* Full WhisperX transcription result object */ },
    "joined_text": "string (Concatenated text from all segments)"
}
```

*   `model_output`: The detailed transcription result from WhisperX, including segments, timestamps, etc.
*   `joined_text`: A single string containing all transcribed text segments joined together.

If an error occurs, a string describing the error is returned.

## Getting Started

### Prerequisites

*   Docker
*   An account on [RunPod](https://runpod.io) if you plan to deploy it.
*   NVIDIA drivers and CUDA toolkit if you intend to build and run with GPU support locally.

### Building the Docker Image

1.  Clone the repository:
    ```bash
    git clone https://github.com/dembrane/runpod-whisper.git
    cd runpod-whisper
    ```

2.  Build the Docker image:
    ```bash
    docker build -t dembrane/runpod-whisper .
    ```

    *   **Note on Gated Models**: If you use a gated model from Hugging Face that requires authentication, you can pass your Hugging Face token as a build argument. The `Dockerfile` includes a commented-out section for downloading model weights during build time:
        ```dockerfile
        # ARG HF_TOKEN
        # ENV HF_TOKEN=${HF_TOKEN}
        # RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='your-gated-model', local_dir='models', local_dir_use_symlinks=False, resume_download=True)"
        ```
        To use this, uncomment the relevant lines in the `Dockerfile` and build with:
        ```bash
        docker build --build-arg HF_TOKEN=your_hf_token -t dembrane/runpod-whisper .
        ```

### Deploying to RunPod

This image is designed for serverless deployment on RunPod.
1.  Push the built Docker image to a container registry (e.g., Docker Hub, Azure Container Registry, GitHub Container Registry).
2.  Create a new Serverless Endpoint on RunPod, pointing to your pushed image.
3.  Configure the environment variables as needed in the RunPod endpoint settings.

## Configuration Details

### Model Selection

The transcription model is specified by the `WHISPER_MODEL_NAME` environment variable. It should be a model compatible with WhisperX (typically faster-whisper models from Hugging Face). The default is `Systran/faster-whisper-large-v1`. Models are downloaded to the `models` directory within the container at runtime if not pre-downloaded during the build.

### Language Support

The `SUPPORTED_LANGUAGES` environment variable takes a comma-separated list of language codes (e.g., "en,es,fr,de,nl"). Tokenizers for these languages are pre-loaded when the worker starts, which can speed up the first request for each of these languages. If a request specifies a language not in this list, the `DEFAULT_LANGUAGE_CODE` (default "en") will be used.

### Compute Resources

The `handler.py` script automatically detects if a CUDA-enabled GPU is available:
*   **GPU**: Uses `cuda` device with `compute_type="float16"`.
*   **CPU**: Uses `cpu` device. If MPS (Apple Silicon) is available, `compute_type="float32"` is used; otherwise, `compute_type="int8"` is used.

The number of CPU threads for the model is set to the available CPU count.

## Local Testing

Use `python handler.py` and it will read inputs from `test_input.json`.

## Dockerfile

The `Dockerfile` sets up the environment for the worker:
*   Base Image: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04` for GPU support.
*   Installs Python 3.10, FFmpeg (static binaries), and other system dependencies.
*   Creates a Python virtual environment and installs dependencies from `requirements.txt` (`whisperx`, `runpod`, `python-dotenv`).
*   Copies the `handler.py` script.
*   Sets the default command to run `python -u handler.py`.
*   Includes a commented-out section to optionally download model weights during the image build process, which can speed up cold starts.

## CI/CD

A GitHub Actions workflow is defined in `.github/workflows/ci.yml`. This workflow:
*   Triggers on pushes to the `main` branch or can be manually dispatched (`workflow_dispatch`).
*   Checks out the repository code.
*   Sets up Docker Buildx for efficient image building.
*   Logs into Azure Container Registry using secrets (`AZURE_REGISTRY_LOGIN_SERVER`, `AZURE_REGISTRY_USERNAME`, `AZURE_REGISTRY_PASSWORD`).
*   Builds the Docker image using the `Dockerfile`.
*   Pushes the built image to the specified Azure Container Registry, tagged with the Git commit SHA (`${{ secrets.AZURE_REGISTRY_LOGIN_SERVER }}/runpod_whisper:${{ github.sha }}`).
*   Utilizes Docker layer caching (from GitHub Actions cache and the registry) to speed up subsequent builds.
