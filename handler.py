import runpod
import os
import time
import whisperx
import gc 
import base64
import tempfile
import requests

# CHECK THE ENV VARIABLES FOR DEVICE AND COMPUTE TYPE
device = os.environ.get('DEVICE', 'cuda') # cpu if on Mac
compute_type = os.environ.get('COMPUTE_TYPE', 'float16') #int8 if on Mac
whisper_model_name = os.environ.get('WHISPER_MODEL_NAME', 'small')
batch_size = 4
default_language_code = "en"

def init():
    """
    Initialize the model.
    """
    global model
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(whisper_model_name, 
                                device, 
                                compute_type=compute_type) # TODO: make this dynamic
    return

def base64_to_tempfile(base64_data):
    """
    Decode base64 data and write it to a temporary file.
    Returns the path to the temporary file.
    """
    # Decode the base64 data to bytes
    audio_data = base64.b64decode(base64_data)

    # Create a temporary file and write the decoded data
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    with open(temp_file.name, 'wb') as file:
        file.write(audio_data)

    return temp_file.name

def download_file(url):
    """
    Download a file from a URL to a temporary file and return its path.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download file from URL")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
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
                    "audio_url": str,
                    "language": str,        # Language code (optional, auto-detects if not provided)
                           # URL of the audio file (optional)
                }
            }
            Either "audio_base_64" or "audio_url" must be provided.
    """
    job_input = event['input']
    job_input_audio_base_64 = job_input.get('audio_base_64')
    job_input_audio_url = job_input.get('audio_url')
    job_input_language = job_input.get('language')

    language_code = job_input_language if job_input_language else default_language_code

    if job_input_audio_base_64:
        audio_input = base64_to_tempfile(job_input_audio_base_64)
    elif job_input_audio_url and job_input_audio_url.startswith('http'):
        audio_input = download_file(job_input_audio_url)
    else:
        return "No audio input provided"
    try:
        # Load the audio
        audio = whisperx.load_audio(audio_input)
        # Transcribe the audio
        result = model.transcribe(audio, batch_size=batch_size, 
        language=language_code, print_progress=True)
        return ' '.join([x['text'] for x in result["segments"]])
    except Exception as e:
        return f"Error transcribing audio: {e}"


init()
runpod.serverless.start({
    "handler": handler
})