import runpod
import whisperx
import base64
import tempfile
import requests
import os
import threading
import torch
from runpod import RunPodLogger
from faster_whisper.tokenizer import Tokenizer
from dataclasses import replace
from langdetect import detect
from litellm import completion
from dotenv import load_dotenv
import concurrent.futures

load_dotenv()
# Global lock to ensure thread-safe calls to detect()
detector_lock = threading.Lock()

logger = RunPodLogger()

if not torch.cuda.is_available() or os.getenv("USE_CPU") == "1":
    logger.info("CUDA is not available")
    logger.info("Using CPU")
    device = "cpu"
    if torch.backends.mps.is_available():
        logger.info("MPS is available. using float32")
        compute_type = "float32"
    else:
        logger.info("MPS is not available. using int8")
        compute_type = "int8"
else:
    logger.info("Using GPU / float16")
    device = "cuda"
    compute_type = "float16"

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "Systran/faster-whisper-large-v1")
TASK = os.getenv("TASK", "transcribe")
DEFAULT_LANGUAGE_CODE = "en"

LITELLM_MODEL = os.getenv("LITELLM_MODEL")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_API_VERSION = os.getenv("LITELLM_API_VERSION")
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE")

USE_LITELLM = False
if LITELLM_MODEL and LITELLM_API_KEY:
    USE_LITELLM = True
    logger.info("Using LiteLLM for translation")


# default to false
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

logger.info(f"Using model: {WHISPER_MODEL_NAME}")

logger.info("Loading the models")

num_threads_available = os.cpu_count()
logger.info(f"threads available: {num_threads_available}")

model = whisperx.load_model(
    WHISPER_MODEL_NAME,
    device=device,
    compute_type=compute_type,
    download_root="models",
    threads=num_threads_available,
)


assert model.tokenizer is None, "Tokenizer is loaded. exit."

logger.info(f"Model loaded: {model}")

# Supported languages
supported_languages = os.getenv("SUPPORTED_LANGUAGES", "en,nl").split(",")

# Prepare tokenizers for all supported languages
tokenizers = {}
for lang in supported_languages:
    logger.info(f"Creating tokenizer for {lang}")
    tokenizers[lang] = Tokenizer(
        model.model.hf_tokenizer,
        # this has to be True, otherwise in Tokenizer self.language_code is set to "en"
        multilingual=True,
        # from load_model they are passed to the tokenizer
        task=TASK,
        language=lang,
    )

logger.info(f"Tokenizers created: {list(tokenizers.keys())}")


def base64_to_tempfile(base64_data):
    logger.debug("Decoding base64 audio data to tempfile.")
    try:
        audio_data = base64.b64decode(base64_data)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        with open(temp_file.name, "wb") as file:
            file.write(audio_data)
        logger.debug(f"Base64 audio written to tempfile: {temp_file.name}")
        return temp_file.name
    except Exception as e:
        logger.error(f"Error in base64_to_tempfile: {e}")
        raise


def download_url_to_mp3(url):
    logger.debug(f"Downloading audio from URL: {url}")
    try:
        response = requests.get(url)
        logger.debug(f"Download response status: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Failed to download file from URL: {url}")
            raise Exception("Failed to download file from URL")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.write(response.content)
        temp_file.close()
        logger.debug(f"Audio downloaded and saved to tempfile: {temp_file.name}")
        return temp_file.name
    except Exception as e:
        logger.error(f"Error in download_url_to_mp3: {e}")
        raise

def translate_text(text, language):
    logger.debug(f"Translating text to {language}. Text: {text}")

    try:
        response = completion(
            model=str(LITELLM_MODEL),
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that translates text to the target language.",
                },
                {
                    "role": "user",
                    "content": f"Translate the following text to {language}: \n\n {text}",
                },
            ],
            api_key=LITELLM_API_KEY,
            api_version=LITELLM_API_VERSION,
            api_base=LITELLM_API_BASE,
        )
        logger.debug(f"Translation response: {response}")
        try:
            return response.choices[0].message.content  # type: ignore
        except Exception as e:
            logger.error(f"Error in translate_text: {e}")
            return text
    except Exception as e:
        logger.error(f"Error in translate_text: {e}")
        raise

def handler(event):
    logger.info(f"Handler called with event: {event}")
    job_input = event["input"]
    job_input_audio_base_64 = job_input.get("audio_base_64")
    job_input_audio_url = job_input.get("audio")
    job_input_language = job_input.get("language", DEFAULT_LANGUAGE_CODE)
    initial_prompt = job_input.get("initial_prompt", "")

    logger.info(f"Job input: {job_input}")
    logger.debug(f"audio_base_64 present: {bool(job_input_audio_base_64)}; audio_url: {job_input_audio_url}; language: {job_input_language}; initial_prompt: {initial_prompt}")

    if job_input_audio_base_64:
        logger.debug("Audio input provided as base64.")
        audio_input = base64_to_tempfile(job_input_audio_base_64)
    elif job_input_audio_url and job_input_audio_url.startswith("http"):
        logger.debug("Audio input provided as URL.")
        audio_input = download_url_to_mp3(job_input_audio_url)
    else:
        logger.error("No audio input provided.")
        return "No audio input provided"

    try:
        if job_input_language not in supported_languages:
            logger.info(
                f"Language {job_input_language} not supported, using default: {DEFAULT_LANGUAGE_CODE}"
            )
            job_input_language = DEFAULT_LANGUAGE_CODE

        logger.info(f"Using language: {job_input_language}")
        tokenizer = tokenizers.get(job_input_language)
        logger.debug(f"Tokenizer for {job_input_language}: {tokenizer}")

        new_options = {
            "initial_prompt": initial_prompt,
        }
        logger.debug(f"Setting model options: {new_options}")
        model.tokenizer = tokenizer
        model.options = replace(model.options, **new_options)

        logger.debug(f"Loading audio from: {audio_input}")
        audio = whisperx.load_audio(audio_input)
        logger.debug("Audio loaded. Running transcription.")

        result = model.transcribe(
            audio,
            task=TASK,
            batch_size=4,
            language=job_input_language,
            print_progress=False,
        )
        logger.debug(f"Transcription result before processing: {result}")

        def process_segment(segment_tuple):
            i, segment = segment_tuple
            segment_text = segment["text"]
            with detector_lock:
                detected_language = detect(segment_text)
            logger.debug(f"Segment {i}: Detected language: {detected_language}, Text: {segment_text}")
            if detected_language != job_input_language:
                logger.info(f"Translating segment {i} from {detected_language} to {job_input_language}")
                segment_text = translate_text(segment_text, job_input_language)
            return (i, segment_text)

        if USE_LITELLM:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads_available
            ) as executor:
                results = executor.map(process_segment, enumerate(result["segments"]))
            sorted_results = sorted(results, key=lambda x: x[0])
            translated_segments = [str(text) for _, text in sorted_results]

            joined_text = " ".join(translated_segments)
            logger.info(f"Joined text: {joined_text}")
        else:
            joined_text = " ".join([segment["text"] for segment in result["segments"]])

        if DEBUG:
            logger.info(f"Full model output: {result}")

        return {
            "model_output": result,
            "joined_text": joined_text,
        }
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return f"Error transcribing audio: {e}"


runpod.serverless.start({"handler": handler})
