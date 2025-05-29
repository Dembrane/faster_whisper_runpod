import runpod
import whisperx
import base64
import tempfile
import requests
import os
import torch
import logging
from faster_whisper.tokenizer import Tokenizer
from dataclasses import replace

logger = logging.getLogger("handler")

if not torch.cuda.is_available():
    logger.info("CUDA is not available")
    logger.info("Using CPU")
    device = "cpu"
    compute_type = "int8"
else:
    logger.info("Using GPU")
    device = "cuda"
    compute_type = "float16"

whisper_model_name = os.getenv(
    "WHISPER_MODEL_NAME", "Systran/faster-distil-whisper-large-v3"
)
task = os.getenv("TASK", "translate")
default_language_code = "en"

logger.info(f"Using model: {whisper_model_name}")


logger.info("Loading the models")

model = whisperx.load_model(
    whisper_model_name,
    device=device,
    compute_type=compute_type,
)

logger.info("Models loaded")

# Supported languages
supported_languages = os.getenv("SUPPORTED_LANGUAGES", "en,nl").split(",")

# Prepare tokenizers for all supported languages
tokenizers = {}
for lang in supported_languages:
    logger.info(f"Creating tokenizer for {lang}")
    tokenizers[lang] = Tokenizer(
        model.model.hf_tokenizer,
        True,
        task=task,
        language=lang,
    )

logger.info("Tokenizers created")


def base64_to_tempfile(base64_data):
    """
    Decode base64 data and write it to a temporary file.
    Returns the path to the temporary file.
    """
    # Decode the base64 data to bytes
    audio_data = base64.b64decode(base64_data)

    # Create a temporary file and write the decoded data
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    with open(temp_file.name, "wb") as file:
        file.write(audio_data)

    return temp_file.name


def download_url_to_mp3(url):
    """
    Download a file from a URL to a temporary file and return its path.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download file from URL")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name


def handler(event):
    """
    Run inference on the model.

    Args:
        event (dict): The input event containing the audio data.
            The event should have the following structure:
            {
                "input": {
                    "audio_base_64": str,  # Base64-encoded audio data (optional)
                    "audio": str,
                    "language": str,        # Language code (optional, auto-detects if not provided)
                    "initial_prompt": str,  # Initial prompt for the model (optional)
                }
            }
            Either "audio_base_64" or "audio" (url) must be provided.
    """
    job_input = event["input"]
    job_input_audio_base_64 = job_input.get("audio_base_64")
    job_input_audio_url = job_input.get("audio")
    job_input_language = job_input.get("language", default_language_code)
    initial_prompt = job_input.get("initial_prompt", "")

    logger.info(f"Job input: {job_input}")

    if job_input_audio_base_64:
        audio_input = base64_to_tempfile(job_input_audio_base_64)
    elif job_input_audio_url and job_input_audio_url.startswith("http"):
        audio_input = download_url_to_mp3(job_input_audio_url)
    else:
        return "No audio input provided"

    try:
        # Use default language if requested language is not supported
        if job_input_language not in supported_languages:
            logger.info(
                f"Language {job_input_language} not supported, using default: {default_language_code}"
            )
            job_input_language = default_language_code

        logger.info(f"Using language: {job_input_language}")
        tokenizer = tokenizers.get(job_input_language)

        new_options = {
            "initial_prompt": initial_prompt,
        }

        model.tokenizer = tokenizer
        model.options = replace(model.options, **new_options)

        # Load the audio
        audio = whisperx.load_audio(audio_input)

        # Transcribe the audio
        result = model.transcribe(
            audio,
            task=task,
            batch_size=8,
            language=job_input_language,
            print_progress=False,
        )

        return {
            "model_output": result,
            "joined_text": " ".join([x["text"] for x in result["segments"]]),
        }
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return f"Error transcribing audio: {e}"


runpod.serverless.start({"handler": handler})
