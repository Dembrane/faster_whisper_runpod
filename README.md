# Dembrane RunPod Whisper

## Overview

The runpod-whisper worker provides fast and accurate audio transcriptions. Built for deployment on RunPod as a serverless endpoint, the worker automatically scales with demand. The worker supports multiple languages, configurable model selection, and runs on CPU or GPU.

## Features

### Core Capabilities

* **Multilingual support** – Supports any language that the configured model supports
* **Hallucination detection** – LLM assigns a hallucination score (0.0–1.0) with reasoning to assess reliability
* **Optional LiteLLM translation** – Automatically translates if not in the requested language
* **Word-level timestamps** – Set `enable_timestamps: true` to include start and end times for each segment

## Environment Variables

Configure the worker behavior through these environment variables:

### Core Configuration

* `WHISPER_MODEL_NAME`: The Hugging Face model name for Faster Whisper
    * Default: `"Systran/faster-whisper-large-v1"`

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

## Handler API Usage

The worker exposes a single handler that expects a JSON payload.

### Input Payload

The `input` object in the JSON payload contains:

```json
{
    "audio_base_64": "string (optional)",
    "audio": "string (URL, optional)",
    "language": "string (optional, e.g., 'en', 'nl')",
    "hotwords": "string (optional)",
    "enable_timestamps": false,
    "metadata_str": "string (optional)",
    "disable_hallucination_detection": false,
    "disable_translation": false
}
```

#### Field Descriptions

* `audio_base_64` (string, optional): Base64 encoded audio data
* `audio` (string, optional): URL / local path pointing to an audio file
* **Note**: Either `audio_base_64` or `audio` must be provided
* `language` (string, optional): Target language code. If the target language is not the same as the detected language, the transcription will be translated to the target language.
* `hotwords` (string, optional): Comma-separated list of hotwords to help transcribe. Use this to inform the model of proper nouns, technical terms, or other words that are important to the conversation.
* `enable_timestamps` (boolean, optional): If `true`, response includes word-level timestamp data
* `metadata_str` (strings, optional): Metadata fields echoed back in response
* `disable_hallucination_detection` (boolean, optional): If `true`, hallucination detection will be disabled
* `disable_translation` (boolean, optional): If `true`, translation will be disabled

### Example Input

```json
{
  "input": {
    "audio": "https://github.com/runpod-workers/sample-inputs/raw/refs/heads/main/audio/Arthur.mp3",
    "language": "nl",
    "hotwords": "RunPod,Directus,Sameer,Dembrane",
    "metadata_str": "This is a test metadata string",
    "enable_timestamps": true,
    "disable_hallucination_detection": false,
    "disable_translation": false
  }
}
```

### Output Payload

#### Success Response

```json
{
    "metadata_str": "optional string",
    "enable_timestamps": true,
    "language": "nl",
    "detected_language": "nl",
    "detected_language_confidence": 0.9805044531822205,
    "joined_text": "... full transcription ...",
    "translation_text": "...full translation...",
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
    "metadata_str": "",
    "enable_timestamps": false,
    "language": "en",
    "error": "No audio input provided",
    "message": "An unhandled error occurred while processing the request."
}
```

## Local Development

1. Clone the repository:
    ```bash
    git clone https://github.com/dembrane/runpod-whisper.git
    cd runpod-whisper
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the environment variables and modify the `test_input.json` file.

4. Run the handler:
    ```bash
    python handler.py
    ```

## Compute Resources

The handler automatically detects available hardware:

* **GPU**: Uses `cuda` device with `compute_type="float16"`
* **CPU with MPS** (Apple Silicon): Uses `cpu` device with `compute_type="float32"`
* **CPU without MPS**: Uses `cpu` device with `compute_type="int8"`

CPU threads are set to the available CPU count.

## Translation Behavior

When LiteLLM is configured: If the detected language is not the same as the requested language, the transcription will be translated to the requested language.


## Hallucination Detection

When LiteLLM is configured, the system analyzes the complete transcription after translation and evaluates for common hallucination patterns:
   * Excessive word/phrase repetition
   * Nonsensical or contradictory sequences
   * Abrupt topic changes
   * Misplaced technical terms
   * Transcribed filler sounds


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


## Security Considerations

* Audio URLs must be publicly accessible
* Temporary files are automatically cleaned up if public audio url is used
* API keys should be stored securely as environment variables
* Consider network policies for production deployments