import runpod
import os
import whisperx
import base64
import tempfile
import requests
import torch
import gc
import logging

logger = logging.getLogger("handler")

# CHECK THE ENV VARIABLES FOR DEVICE AND COMPUTE TYPE
device = os.environ.get("DEVICE", "cuda")  # cpu if on Mac
compute_type = os.environ.get("COMPUTE_TYPE", "float16")  # int8 if on Mac

whisper_model_name = "deepdml/faster-whisper-large-v3-turbo-ct2"
batch_size = 1
default_language_code = "en"

# Download the model
logger.info("Downloading the model")
preload_model = whisperx.load_model(
    whisper_model_name,
    device,
    compute_type=compute_type,
    download_root="models",
)
logger.info("Model downloaded")

logger.info("Cleaning up")
del preload_model
torch.cuda.empty_cache()
gc.collect()
logger.info("Cleaned up")

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


def download_file(url):
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
            Either "audio_base_64" or "audio_url" must be provided.
    """
    job_input = event["input"]
    job_input_audio_base_64 = job_input.get("audio_base_64")
    job_input_audio_url = job_input.get("audio")
    job_input_language = job_input.get("language")

    language_code = job_input_language if job_input_language else default_language_code

    if job_input_audio_base_64:
        audio_input = base64_to_tempfile(job_input_audio_base_64)
    elif job_input_audio_url and job_input_audio_url.startswith("http"):
        audio_input = download_file(job_input_audio_url)
    else:
        return "No audio input provided"
    try:
        model = whisperx.load_model(
            whisper_model_name,
            device,
            compute_type=compute_type,
            local_files_only=True,
            download_root="models",
            asr_options={
                "initial_prompt": job_input.get("initial_prompt", ""),
            },
        )

        logger.info(f"Model loaded: {model}")

        # Load the audio
        audio = whisperx.load_audio(audio_input)

        # Transcribe the audio
        result = model.transcribe(
            audio, batch_size=batch_size, language=language_code, print_progress=True
        )

        return {
            "model_output": result,
            "joined_text": " ".join([x["text"] for x in result["segments"]]),
        }
    except Exception as e:
        return f"Error transcribing audio: {e}"


runpod.serverless.start({"handler": handler})
