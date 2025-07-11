# Dembrane RunPod Whisper

## Overview

The runpod-whisper worker uses WhisperX (which uses faster-whisper) to provide fast and accurate audio transcriptions. Built for deployment on RunPod as a serverless endpoint, the worker automatically scales with demand. The worker supports multiple languages, configurable model selection, and runs on CPU or GPU.

## Features

### Core Capabilities

* **Fast WhisperX transcription** – Low-latency, accurate speech-to-text conversion
* **Optional LiteLLM translation** – Automatically translates segments not in the requested language
* **Hallucination detection** – LLM assigns a hallucination score (0.0–1.0) with reasoning to assess reliability
* **Segment-level timestamps** – Set `enable_timestamps: true` to include start and end times for each segment
* **Thread-safe language detection** – Reliable detection of segment languages for translation decisions

### Additional Features

* **Flexible audio input** – Accept base64 blobs or remote URLs
* **Configurable model and multi-language support** – Choose any WhisperX-compatible model and preload tokenizers
* **Optimized for RunPod with automatic hardware detection** – Seamlessly uses GPU (float16) or CPU (int8/float32)
* **Automatic cleanup** – Temporary files are removed after processing
* **Comprehensive error handling** – Detailed error reporting with metadata preservation

## Environment Variables

Configure the worker behavior through these environment variables:

### Core Configuration

* `WHISPER_MODEL_NAME`: The Hugging Face model name for WhisperX
    * Default: `"Systran/faster-whisper-large-v1"`
* `TASK`: The Whisper task type
    * Default: `"transcribe"`
* `DEFAULT_LANGUAGE_CODE`: Default language code when none specified in request
    * Default: `"en"`
* `SUPPORTED_LANGUAGES`: Comma-separated list of language codes for pre-loaded tokenizers
    * Default: `"en,nl"`
    * Example: `"en,nl,es,fr,de"`

### Performance Settings

* `BATCH_SIZE`: Number of audio segments processed per batch
    * Default: `"8"`
* `USE_CPU`: Force CPU execution even when GPU is available
    * Default: Not set (auto-detect)
    * Set to `"1"` to force CPU usage
* `DEBUG`: Enable detailed debug logging
    * Default: `"false"`
    * Set to `"true"`, `"1"`, or `"yes"` to enable

### LiteLLM Integration

Configure these variables to enable translation and hallucination detection:

* `LITELLM_MODEL`: LiteLLM model identifier (required for LiteLLM features)
* `LITELLM_API_KEY`: API key for LiteLLM service (required)
* `LITELLM_API_BASE`: Base URL for LiteLLM API (optional)
* `LITELLM_API_VERSION`: API version for LiteLLM (optional)

## Handler API

The worker exposes a single handler that expects a JSON payload.

### Input Payload

The `input` object in the JSON payload contains:

```json
{
    "audio_base_64": "string (optional)",
    "audio": "string (URL, optional)",
    "language": "string (optional, e.g., 'en', 'nl')",
    "initial_prompt": "string (optional)",
    "enable_timestamps": false,
    "conversation_id": "string (optional)",
    "conversation_chunk_id": "string (optional)",
    "metadata_str": "string (optional)"
}
```

#### Field Descriptions

* `audio_base_64` (string, optional): Base64 encoded audio data
* `audio` (string, optional): URL pointing to an audio file (must start with "http")
* **Note**: Either `audio_base_64` or `audio` must be provided
* `language` (string, optional): Target language code. Defaults to `DEFAULT_LANGUAGE_CODE` if missing or unsupported
* `initial_prompt` (string, optional): Initial prompt to guide transcription and provide context for hallucination detection
* `enable_timestamps` (boolean, optional): If `true`, response includes segment-level timestamp data
* `conversation_id`, `conversation_chunk_id`, `metadata_str` (strings, optional): Metadata fields echoed back in response

### Example Input

```json
{
    "input": {
        "audio": "https://example.com/audio.mp3",
        "language": "nl",
        "initial_prompt": "Hallo, laten we beginnen. Eerst even een introductieronde.",
        "enable_timestamps": true,
        "conversation_id": "123",
        "conversation_chunk_id": "456"
    }
}
```

### Output Payload

#### Success Response

```json
{
    "conversation_id": "123",
    "conversation_chunk_id": "456",
    "metadata_str": "optional string",
    "enable_timestamps": true,
    "language": "nl",
    "joined_text": "... full transcription ...",
    "translation_error": false,
    "hallucination_score": 0.2,
    "hallucination_reason": "Minor repetitions detected",
    "segments": [
        {
            "text": "Segment text",
            "start": 0.0,
            "end": 2.5
        }
    ]
}
```

#### Response Fields

* `joined_text`: Complete transcription text (translated if needed)
* `translation_error`: `true` if any translation failed or timed out
* `hallucination_score`: Float 0.0–1.0 indicating severity:
    * 0.0: No hallucination detected
    * 0.1–0.3: Minor errors, meaning intact
    * 0.4–0.6: Moderate errors, partial distortion
    * 0.7–0.9: Severe errors, strong distortion
    * 1.0: Complete hallucination/nonsense
* `hallucination_reason`: Brief explanation (max 20 words) when score > 0
* `segments`: Array of segment objects (only when `enable_timestamps: true`)

#### Error Response

```json
{
    "conversation_id": "123",
    "conversation_chunk_id": "456",
    "metadata_str": "",
    "enable_timestamps": false,
    "language": "en",
    "error": "No audio input provided",
    "message": "An unhandled error occurred while processing the request."
}
```

## Getting Started

### Prerequisites

* Docker
* RunPod account for deployment
* NVIDIA drivers and CUDA toolkit for local GPU testing

### Building the Docker Image

1. Clone the repository:
    ```bash
    git clone https://github.com/dembrane/runpod-whisper.git
    cd runpod-whisper
    ```

2. Build the Docker image:
    ```bash
    docker build -t dembrane/runpod-whisper .
    ```

### Using Gated Models

For Hugging Face gated models requiring authentication:

1. Uncomment the relevant lines in the Dockerfile
2. Build with your HF token:
    ```bash
    docker build --build-arg HF_TOKEN=your_hf_token -t dembrane/runpod-whisper .
    ```

### Deploying to RunPod

1. Push the Docker image to a container registry (Docker Hub, Azure Container Registry, GitHub Container Registry)
2. Create a new Serverless Endpoint on RunPod using your image
3. Configure environment variables in RunPod endpoint settings

## Configuration Details

### Model Selection

The transcription model is specified by `WHISPER_MODEL_NAME`. Use any WhisperX-compatible model from Hugging Face. Models download to the `models` directory at runtime unless pre-downloaded during build.

### Language Support

The `SUPPORTED_LANGUAGES` variable accepts comma-separated language codes (e.g., "en,es,fr,de,nl"). Tokenizers for these languages load at startup, improving first-request performance. Requests for unsupported languages fall back to `DEFAULT_LANGUAGE_CODE`.

### Compute Resources

The handler automatically detects available hardware:

* **GPU**: Uses `cuda` device with `compute_type="float16"`
* **CPU with MPS** (Apple Silicon): Uses `cpu` device with `compute_type="float32"`
* **CPU without MPS**: Uses `cpu` device with `compute_type="int8"`

CPU threads are set to the available CPU count.

### Translation Behavior

When LiteLLM is configured:

1. Each segment's language is detected using `langdetect`
2. Segments not in the target language are buffered
3. Contiguous non-target segments are translated together for efficiency
4. Translation preserves proper nouns and technical terms
5. Translation timeout is 10 seconds per request

### Hallucination Detection

When LiteLLM is configured, the system:

1. Analyzes the complete transcription after translation
2. Evaluates for common hallucination patterns:
   * Excessive word/phrase repetition
   * Nonsensical or contradictory sequences
   * Abrupt topic changes
   * Misplaced technical terms
   * Transcribed filler sounds
3. Returns a score and brief explanation
4. Uses the initial prompt for context when available

## Local Testing

Run locally with test input:

```bash
python handler.py
```

The handler reads from `test_input.json` in the current directory.

## Dockerfile

The Dockerfile configures the environment:

* **Base Image**: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04` for GPU support
* **Dependencies**: Python 3.10, FFmpeg (static binaries), system libraries
* **Python Environment**: Virtual environment with dependencies from `requirements.txt`
* **Entry Point**: `python -u handler.py`

### Key Dependencies

* `whisperx`: Core transcription engine
* `runpod`: Serverless framework
* `langdetect`: Language detection for translation routing
* `litellm`: LLM integration for translation and hallucination detection
* `torch`: PyTorch for model execution
* `python-dotenv`: Environment variable management

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):

* **Triggers**: Push to `main` branch or manual dispatch
* **Actions**:
    * Builds Docker image
    * Pushes to Azure Container Registry
    * Tags with Git commit SHA
    * Uses layer caching for efficiency

### Required Secrets

* `AZURE_REGISTRY_LOGIN_SERVER`
* `AZURE_REGISTRY_USERNAME`
* `AZURE_REGISTRY_PASSWORD`

## Performance Considerations

* **Cold Start**: Initial model loading takes 10-30 seconds depending on model size
* **Batch Processing**: Adjust `BATCH_SIZE` based on available memory
* **Translation**: Adds latency when segments need translation
* **Hallucination Detection**: Adds 1-3 seconds for LLM analysis

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `BATCH_SIZE` or use smaller model
2. **Translation Timeouts**: Check LiteLLM configuration and API limits
3. **Unsupported Language**: Verify language code in `SUPPORTED_LANGUAGES`
4. **Audio Download Fails**: Ensure URL is accessible and returns valid audio

### Debug Mode

Enable debug logging to troubleshoot:

```bash
DEBUG=true
```

## Security Considerations

* Audio URLs must be publicly accessible
* Temporary files are automatically cleaned up
* API keys should be stored securely as environment variables
* Consider network policies for production deployments