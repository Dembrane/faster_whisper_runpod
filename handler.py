import runpod
import base64
import tempfile
import requests
import os
import enum
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.audio import decode_audio
from runpod import RunPodLogger
from litellm import completion
from litellm.exceptions import Timeout
from dotenv import load_dotenv
import traceback
import logging

load_dotenv(verbose=True, override=True)

logger = RunPodLogger()

# default to false
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
logger.info(f"Debug mode: {DEBUG}")

if DEBUG:
	logging.getLogger("faster_whisper").setLevel(logging.DEBUG)


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


# number of segments to process at once
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
logger.info(f"Using batch size: {BATCH_SIZE}")

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "Systran/faster-whisper-large-v3")
logger.info(f"Using model: {WHISPER_MODEL_NAME}")

# translate - convert to english
# transcribe - output in target language
TASK = os.getenv("TASK", "transcribe")
DEFAULT_LANGUAGE_CODE = "en"

USE_LITELLM = False

LITELLM_MODEL = os.getenv("LITELLM_MODEL")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_API_VERSION = os.getenv("LITELLM_API_VERSION")
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE")


if LITELLM_MODEL and LITELLM_API_KEY:
	USE_LITELLM = True
	logger.info("Using LiteLLM for translation")


num_threads_available = os.cpu_count()
logger.info(f"threads available: {num_threads_available}")

# model = whisperx.load_model(
# 	WHISPER_MODEL_NAME,
# 	device=device,
# 	compute_type=compute_type,
# 	download_root="models",
# 	threads=num_threads_available,
# )

model = WhisperModel(
	WHISPER_MODEL_NAME,
	device=device,
	compute_type=compute_type,
	download_root="models",
)

batched_model = BatchedInferencePipeline(
	model,
)


logger.info(f"Model loaded: {model}")

# Supported languages
supported_languages = os.getenv("SUPPORTED_LANGUAGES", "en,nl").split(",")

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

class ExitCode(enum.Enum):
	SUCCESS = 0
	ERROR_TRANSLATING = 1
	TIMEOUT = 2


def translate_text(text, language) -> tuple[str, int]:
	logger.debug(f"Translating text to {language}. Text: {text}")

	try:
		system_prompt = (
			"""You are a helpful assistant that translates text to the target language.\n"
			"Important guidelines:\n"
			"- Preserve all proper nouns exactly as they appear\n"
			"- Maintain the original meaning and context\n"
			"- If you encounter technical terms or abbreviations, keep them unchanged\n"
			"- Do not add explanations or commentary to the translation"""
		)

		response = completion(
			model=str(LITELLM_MODEL),
			messages=[
				{"role": "system", "content": system_prompt},
				{
					"role": "user",
					"content": f"Translate the following text to {language}. Only return the translation, nothing else: \n\n{text}",
				},
			],
			api_key=LITELLM_API_KEY,
			api_version=LITELLM_API_VERSION,
			api_base=LITELLM_API_BASE,
			timeout=10
		)
		logger.debug(f"Translation response: {response}")
		return response.choices[0].message.content, ExitCode.SUCCESS.value  # type: ignore
	except Timeout as e:
		logger.error(f"Timeout in translate_text: {e}")
		return text, ExitCode.TIMEOUT.value
	except Exception as e:
		logger.error(f"Error in translate_text: {e}")
		return text, ExitCode.ERROR_TRANSLATING.value


# -----------------------------------
# Hallucination Detection
# -----------------------------------

HALLUCINATION_PROMPT = """You are an expert QA analyst specializing in reviewing speech-to-text (ASR) transcripts. Your task is to identify "hallucinations" without access to the original audio.

A hallucination is defined as text that is highly unlikely to have been spoken by a coherent human. You must infer this from textual evidence alone.

**Crucially, distinguish between:**
- **Hallucinations (High Penalty):** Invented phrases, nonsensical "word salad," illogical topic shifts, or repetitive loops that suggest ASR model failure.
- **Minor Inaccuracies (Low/No Penalty):** Simple transcription errors (e.g., "their" vs "there"), misheard names, or the transcription of natural filler words (e.g., "um," "uh," "like"). Do not score these as severe hallucinations.

**Evaluation Criteria (assign ONE score based on the MOST SEVERE issue found):**
- **0.0 - No Hallucination:** The text is coherent and sounds like natural human speech or narration.
- **0.1-0.3 - Minor:** Mostly coherent but contains a few slightly awkward or out-of-place words that make a phrase sound unnatural. Overall meaning is clear. (e.g., "I feel like can happen towards a link.")
- **0.4-0.6 - Moderate:** Contains distracting nonsensical phrases or confusing sentences that disrupt the flow and partially obscure the meaning. (e.g., "you can't wait for our students" in a context that makes no sense.)
- **0.7-0.9 - Severe:** Multiple incoherent passages, significant logical breaks, or repetitive loops. The core meaning is heavily obscured.
- **1.0 - Complete:** The text is almost entirely gibberish, nonsensical, or stuck in a repetitive loop.

**Primary Hallucination Signals to Look For:**
1.  **Logical Incoherence or Nonsense:** Sentences or phrases that are grammatically malformed, self-contradictory, or simply make no sense.
2.  **Semantic Repetition or Looping:** The same idea or phrase repeated unnaturally multiple times in a row.
3.  **Abrupt and Illogical Topic Shifts:** Sudden changes in topic that lack any logical transition, suggesting the ASR model lost track.
4.  **Inappropriate Jargon or "Word Salad":** Use of technical or specific terms that are completely out of context.

Respond ONLY with a valid JSON object in the following format:
{
  "hallucination_score": <score from 0.0 to 1.0>,
  "reason": "<A concise, max 20-word explanation for your score, citing the most significant issue.>"
}
"""


def detect_hallucination(text: str) -> tuple[float, str]:
	"""Detect hallucinations in text using LLM analysis and return a score between 0 (no hallucination) and 1 (severe hallucination)."""
	logger.debug(f"Checking for hallucinations in text: {text[:100]}...")

	if not (LITELLM_MODEL and LITELLM_API_KEY):
		# LiteLLM not configured; skip detection
		return 0.0, ""

	try:
		response = completion(
			model=str(LITELLM_MODEL),
			messages=[
				{
					"role": "system",
					"content": HALLUCINATION_PROMPT,
				},
				{
					"role": "user",
					"content": f"Analyze this ASR transcript for signs of hallucination. Transcript: {text}",
				},
			],
			api_key=LITELLM_API_KEY,
			api_version=LITELLM_API_VERSION,
			api_base=LITELLM_API_BASE,
			response_format={"type": "json_object"},
			timeout=10
		)

		result = response.choices[0].message.content
		import json
		parsed = json.loads(result)
		score = parsed.get("hallucination_score", 0.0)
		# Ensure score is a float between 0 and 1 in 0.1 increments
		try:
			score = float(score)
		except (TypeError, ValueError):
			score = 0.0
		return score, parsed.get("reason", "")
	except Exception as e:
		logger.error(f"Error in hallucination detection: {e}")
		return 0.0, f"Detection error: {str(e)}"


def handler(event):
	logger.debug(f"Handler called with event: {event}")

	try:
		# ---------------------- Parse Input ----------------------
		job_input = event["input"]
		job_input_audio_base_64 = job_input.get("audio_base_64")
		job_input_audio_url = job_input.get("audio")
		job_input_language = job_input.get("language", DEFAULT_LANGUAGE_CODE)
		initial_prompt = job_input.get("initial_prompt", None)
		hotwords = job_input.get("hotwords", None)
		enable_timestamps = job_input.get("enable_timestamps", False)

		# Metadata
		conversation_id = job_input.get("conversation_id", "")
		conversation_chunk_id = job_input.get("conversation_chunk_id", "")
		metadata_str = job_input.get("metadata_str", "")

		logger.info(f"Job input: {job_input}")
		logger.debug(
			f"audio_base_64 present: {bool(job_input_audio_base_64)}; audio_url: {job_input_audio_url}; language: {job_input_language}; initial_prompt: {initial_prompt}"
		)

		# ---------------------- Audio Preparation ----------------------
		if job_input_audio_base_64:
			logger.debug("Audio input provided as base64.")
			audio_input = base64_to_tempfile(job_input_audio_base_64)
		elif job_input_audio_url and job_input_audio_url.startswith("http"):
			logger.debug("Audio input provided as URL.")
			audio_input = download_url_to_mp3(job_input_audio_url)
		else:
			logger.error("No audio input provided.")
			raise ValueError("No audio input provided")
		
		# ---------------------- Language & Tokenizer ----------------------
		if job_input_language not in supported_languages:
			logger.info(
				f"Language {job_input_language} not supported, using default: {DEFAULT_LANGUAGE_CODE}"
			)
			job_input_language = DEFAULT_LANGUAGE_CODE

		logger.debug(f"Loading audio from: {audio_input}")
		audio = decode_audio(audio_input)
		logger.debug("Audio loaded. Running transcription.")

		segments, info = batched_model.transcribe(
			audio=audio,
			# task: Task to perform. "transcribe" transcribes the audio to text. "translate" transcribes the audio to text and translates it to the target language.
			task=TASK,
			# hotwords: Hotwords/hint phrases to the model. Has no effect if prefix is not None.
			hotwords=hotwords,
			# language: language: The language spoken in the audio. It should be a language code such as "en" or "fr". If not set, the language will be detected in the first 30 seconds of audio.
			language=job_input_language,
			word_timestamps=enable_timestamps,
			# multilingual: Perform language detection on every segment.
			multilingual=True,
			batch_size=BATCH_SIZE,
			log_progress=DEBUG,
			initial_prompt=initial_prompt,
		)


		generated_segments = []

		for seg_el in segments:
			words = []
			for word_el in seg_el.words:
				words.append({
					"word": word_el.word,
					"start": word_el.start,
					"end": word_el.end,
				})
			
			append_obj = {
				"text": seg_el.text,
				"words": words,
				"start": seg_el.start,
				"end": seg_el.end,
			}
			 
			generated_segments.append(append_obj)
			
			if DEBUG:
				logger.debug(f"decoded segment: {append_obj}")


		joined_text = " ".join([segment["text"] for segment in generated_segments])

		translation_error_occurred = False
		hallucination_score = 0.0
		hallucination_reason = ""

		if USE_LITELLM:
			if joined_text:
				hallucination_score, hallucination_reason = detect_hallucination(joined_text)
				if hallucination_score >= 0.9:
					logger.info(f"Severe hallucination detected (score={hallucination_score}): {hallucination_reason}")
				elif hallucination_score >= 0.5:
					logger.info(f"Moderate hallucination detected (score={hallucination_score}): {hallucination_reason}")

		result = {
			# metadata
			"conversation_id": conversation_id,
			"conversation_chunk_id": conversation_chunk_id,
			"metadata_str": metadata_str,
			"enable_timestamps": enable_timestamps,
			"language": job_input_language,
			"whisper_language": info.language_probability,
			"whisper_language_probability": info.language_probability,
			"translation_error": translation_error_occurred,
			"hallucination_score": hallucination_score,
			"hallucination_reason": hallucination_reason if hallucination_score > 0 else "",
			"joined_text": joined_text,
		}

		try:
			if audio_input:
				os.remove(audio_input)
		except Exception as e:
			logger.error(f"Error in cleanup: {e}")

		if enable_timestamps:
			result["segments"] = generated_segments

		if DEBUG:
			import json
			with open("/tmp/result.json", "w") as f:
				json.dump(result, f)

		return result

	# ---------------------- Global Exception Handler ----------------------
	except Exception as e:
		logger.error(f"Unhandled error: {str(e)}")
		logger.error(traceback.format_exc())

		try:
			if audio_input:
				os.remove(audio_input)
		except Exception as inner_e:
			logger.error(f"Error in cleanup: {inner_e}")

		# Build minimal common metadata if available
		common_meta = {
			"conversation_id": locals().get("conversation_id", ""),
			"conversation_chunk_id": locals().get("conversation_chunk_id", ""),
			"metadata_str": locals().get("metadata_str", ""),
			"enable_timestamps": locals().get("enable_timestamps", False),
			"language": locals().get("job_input_language", DEFAULT_LANGUAGE_CODE),
		}

		# Return a generic error payload to the caller (includes metadata)
		return {
			**common_meta,
			"error": str(e),
			"message": "An unhandled error occurred while processing the request.",
		}


runpod.serverless.start({"handler": handler})
