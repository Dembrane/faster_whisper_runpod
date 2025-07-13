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

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "Systran/faster-whisper-large-v1")
logger.info(f"Using model: {WHISPER_MODEL_NAME}")

USE_LITELLM = False

LITELLM_MODEL = os.getenv("LITELLM_MODEL")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_API_VERSION = os.getenv("LITELLM_API_VERSION")
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE")

if LITELLM_MODEL and LITELLM_API_KEY:
	USE_LITELLM = True
	logger.info("Using LiteLLM for translation")


model = WhisperModel(
	WHISPER_MODEL_NAME,
	device=device,
	compute_type=compute_type,
	download_root="models",
)

batched_model = BatchedInferencePipeline(
	model,
)

logger.info(f"Model loaded: {WHISPER_MODEL_NAME}")


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


# This can be defined as a constant
TRANSLATION_PROMPT = (
    """You are an expert multilingual translation assistant. Your task is to produce a single, coherent text in the specified target language, ensuring the final output is natural and grammatically correct.

**Your Process:**
1.  **Identify:** Pinpoint the phrases in the source text that are NOT in the target language.
2.  **Translate:** Translate ONLY those identified parts into the target language.
3.  **Preserve:** Leave any text that is ALREADY in the target language completely unchanged.
4.  **Integrate:** Combine the newly translated parts and the preserved parts into a single, seamless output text.

**CRITICAL GUIDELINES:**
- **Hotwords:** The user has provided a list of "hotwords". You MUST preserve these words exactly as they appear, including their original capitalization. DO NOT translate them.
- **Output:** Output ONLY the final integrated text. Do not add commentary, explanations, or any other text.

---
**Example 1: Mixed Language with Hotwords**
- **Target Language:** `nl`
- **Hotwords:** ["RunPod", "WhisperX"]
- **Source Text:** `I am running the new WhisperX model on the RunPod server. Ik deel nu mijn scherm.`
- **Your Correct Output:** `Ik draai het nieuwe WhisperX model op de RunPod server. Ik deel nu mijn scherm.`
---
**Example 2: Simple Translation with Hotwords**
- **Target Language:** `es`
- **Hotwords:** ["Project Phoenix"]
- **Source Text:** `Let's discuss the status of Project Phoenix.`
- **Your Correct Output:** `Discutamos el estado de Project Phoenix.`
---
"""
)

def translate_text(
    text: str,
    language: str,
    hotwords: str = None,
) -> tuple[str, int]:
    """
    Translates text to the target language, preserving specified hotwords.

    Args:
        text: The source text to translate.
        language: The 2-letter target language code (e.g., 'nl', 'es').
        hotwords: An optional list of words/phrases to preserve exactly.

    Returns:
        A tuple containing the translated text and an exit code.
    """
    logger.debug(f"Translating text to '{language}' with hotwords: {hotwords}. Text: {text[:100]}...")

    user_prompt_parts = [
        f"Target Language: `{language}`"
    ]

    if hotwords:
        user_prompt_parts.append(f"Hotwords: [{hotwords}]")

    user_prompt_parts.append(f'\nSource Text: "{text}"')

    user_prompt = "\n".join(user_prompt_parts)

    try:
        response = completion(
            model=str(LITELLM_MODEL),
            messages=[
                {"role": "system", "content": TRANSLATION_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            api_key=LITELLM_API_KEY,
            api_version=LITELLM_API_VERSION,
            api_base=LITELLM_API_BASE,
            timeout=20
        )
        translated_text = response.choices[0].message.content
        logger.debug(f"Translation response: {translated_text}")
        return translated_text, ExitCode.SUCCESS.value

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
- **Minor Inaccuracies (Low/No Penalty):** Simple transcription errors, misheard names, or natural filler words (e.g., "um," "uh"). Do not score these as severe hallucinations.

**Evaluation Criteria (assign ONE score based on the MOST SEVERE issue found):**
- **0.0 - No Hallucination:** The text is coherent and sounds like natural human speech.
- **0.1-0.3 - Minor:** Mostly coherent but contains a few slightly awkward or out-of-place words.
- **0.4-0.5 - Moderate:** Contains distracting nonsensical phrases or confusing sentences that disrupt the flow.
- **0.6-0.9 - Severe:** Multiple incoherent passages, significant logical breaks, or repetitive loops that obscure meaning.
- **1.0 - Complete:** The text is almost entirely gibberish, stuck in a repetitive loop, OR consists solely of a list of the provided "Hotwords" with no surrounding conversational context.

**Special Case: Isolated Keyword (Hotword) Lists**
If the ASR output consists ONLY of a list of the provided "Hotwords" (e.g., just the names themselves, separated by commas or spaces) and nothing else, this is a sign of complete ASR failure. **This case MUST be scored as 1.0.**

**Primary Hallucination Signals to Look For:**
1.  **Logical Incoherence or Nonsense:** Grammatically malformed or self-contradictory sentences.
2.  **Semantic Repetition or Looping:** The same idea or phrase repeated unnaturally.
3.  **Abrupt and Illogical Topic Shifts:** Sudden changes in topic without logical transition.
4.  **Inappropriate Jargon or "Word Salad":** Technical terms used completely out of context.

Respond ONLY with a valid JSON object in the following format:
{
  "hallucination_score": <score from 0.0 to 1.0>,
  "reason": "<A concise, max 20-word explanation for your score, citing the most significant issue.>"
}
"""


def detect_hallucination(text: str, hotwords: str = "") -> tuple[float, str]:
    """Detect hallucinations in text using LLM analysis and return a score between 0 (no hallucination) and 1 (severe hallucination)."""
    logger.debug(f"Checking for hallucinations in text: {text[:100]}...")

    if not (LITELLM_MODEL and LITELLM_API_KEY):
        # LiteLLM not configured; skip detection
        return 0.0, ""

    try:
        # 2. Conditionally create the hotwords info string
        hotwords_info = f"Hotwords: {hotwords}\n" if hotwords else ""

        # 3. Update the user content to include the hotwords_info
        user_content = (
            f"Analyze this ASR transcript for signs of hallucination.\n"
            f"{hotwords_info}\n"
            f"Transcript: {text}"
        )

        response = completion(
            model=str(LITELLM_MODEL),
            messages=[
                {
                    "role": "system",
                    "content": HALLUCINATION_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_content, # Use the new user_content string
                },
            ],
            api_key=LITELLM_API_KEY,
            api_version=LITELLM_API_VERSION,
            api_base=LITELLM_API_BASE,
            response_format={"type": "json_object"},
            timeout=10
        )
        # ... (rest of the function remains the same)
        result = response.choices[0].message.content
        import json
        parsed = json.loads(result)
        score = parsed.get("hallucination_score", 0.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        return score, parsed.get("reason", "")
    except Exception as e:
        logger.error(f"Error in hallucination detection: {e}")
        return 0.0, f"Detection error: {str(e)}"


def clean_up_audio(audio_input = None):
	try:
		if audio_input:
			if not DEBUG:
				logger.debug(f"Removing audio input: {audio_input}")
				os.remove(audio_input)
			else:
				logger.debug("not removing audio input (DEBUG mode)")
	except Exception as e:
		logger.error(f"Error in clean_up_audio: {e}")


def handler(event):
	logger.debug(f"Handler called with event: {event}")

	try:
		# ---------------------- Parse Input ----------------------
		job_input = event["input"]
		job_input_audio_base_64 = job_input.get("audio_base_64")
		job_input_audio_url = job_input.get("audio")
		job_input_language = job_input.get("language", None)
		initial_prompt = job_input.get("initial_prompt", None)
		if initial_prompt and len(initial_prompt) > 0:
			logger.warning("initial_prompt is deprecated and will be removed in a future version. Please use the hotwords parameter instead.")
		hotwords = job_input.get("hotwords", None)
		enable_timestamps = job_input.get("enable_timestamps", False)
		disable_hallucination_detection = job_input.get(
			"disable_hallucination_detection", False
		)
		disable_translation = job_input.get("disable_translation", False)

		# Metadata
		conversation_id = job_input.get("conversation_id", "")
		conversation_chunk_id = job_input.get("conversation_chunk_id", "")
		metadata_str = job_input.get("metadata_str", "")

		logger.info(f"Job input: {job_input}")

		# ---------------------- Audio Preparation ----------------------
		if job_input_audio_base_64:
			logger.debug("Audio input provided as base64.")
			audio_input = base64_to_tempfile(job_input_audio_base_64)
		elif job_input_audio_url:
			if job_input_audio_url.startswith("http"):
				logger.debug("Audio input provided as URL.")
				audio_input = download_url_to_mp3(job_input_audio_url)
			else:
				logger.debug("Audio input provided as local file.")
				if not os.path.exists(job_input_audio_url):
					logger.error(f"Local file {job_input_audio_url} does not exist.")
					raise ValueError(
						f"Local file {job_input_audio_url} does not exist."
					)
				audio_input = job_input_audio_url
		else:
			logger.error("No audio input provided.")
			raise ValueError("No audio input provided")

		logger.debug(f"Loading audio from: {audio_input}")
		audio = decode_audio(audio_input)
		logger.debug("Audio loaded. Running transcription.")

		segments, info = batched_model.transcribe(
			audio=audio,
			# task: Task to perform. "transcribe" transcribes the audio to text. "translate" transcribes the audio to text and translates it to the target language.
			task="transcribe",
			# hotwords: Hotwords/hint phrases to the model. Has no effect if prefix is not None.
			hotwords=hotwords,
			# language: language: The language spoken in the audio. It should be a language code such as "en" or "fr". If not set, the language will be detected in the first 30 seconds of audio.
			# language=model_language,
			word_timestamps=enable_timestamps,
			# multilingual: Perform language detection on every segment.
			# multilingual=False,
			batch_size=BATCH_SIZE,
			log_progress=DEBUG,
		)

		generated_segments = []

		if segments:
			for seg_el in segments:
				words = None
				if seg_el.words:
					words = []
					for word_el in seg_el.words:
						words.append(
							{
								"word": word_el.word,
								"start": word_el.start,
								"end": word_el.end,
							}
						)

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

		# ---------------------- Translation ----------------------
		translation_text = None
		translation_error = None

		needs_translation = (
			not disable_translation and
			job_input_language and
			info.language != job_input_language
		)

		if needs_translation:
			logger.debug(f"Translating text to '{job_input_language}' with hotwords: {hotwords}. Text: {joined_text[:100]}...")
			translation_text, translation_error = translate_text(
				joined_text,
				job_input_language,
				hotwords
			)
		else:
			if disable_translation:
				logger.debug("Translation not needed because disable_translation is True")
			elif not job_input_language:
				logger.debug("Translation not needed because no language was provided")
			else:
				logger.debug(f"Translation not needed because detected language matches input language. \n({info.language}-{info.language_probability} / desired {job_input_language})")

		# ---------------------- Hallucination Detection ----------------------
		hallucination_score = None
		hallucination_reason = None

		if USE_LITELLM and not disable_hallucination_detection:
			text_to_analyze_for_hallucination = translation_text if translation_text else joined_text
			if text_to_analyze_for_hallucination:
				hallucination_score, hallucination_reason = detect_hallucination(
					text_to_analyze_for_hallucination,
					hotwords
				)
				if hallucination_score >= 0.9:
					logger.info(
						f"Severe hallucination detected (score={hallucination_score}): {hallucination_reason}"
					)
				elif hallucination_score >= 0.5:
					logger.info(
						f"Moderate hallucination detected (score={hallucination_score}): {hallucination_reason}"
					)

		result = {
			# metadata
			"conversation_id": conversation_id,
			"conversation_chunk_id": conversation_chunk_id,
			"metadata_str": metadata_str,
			"enable_timestamps": enable_timestamps,
			"language": job_input_language,
			"detected_language": info.language,
			"detected_language_confidence": info.language_probability,
			"translation_text": translation_text,
			"translation_error": translation_error,
			"hallucination_score": hallucination_score,
			"hallucination_reason": hallucination_reason
				if hallucination_score is not None
				else "",
			
			"joined_text": joined_text,
		}

		clean_up_audio(audio_input)

		if enable_timestamps:
			result["segments"] = generated_segments
		

		if DEBUG:
			import json

			with open(
				"/Users/sameerpashikantidembrane/dev/runpod-whisper/results.json", "w"
			) as f:
				json.dump(result, f)

		return result

	# ---------------------- Global Exception Handler ----------------------
	except Exception as e:
		logger.error(f"Unhandled error: {str(e)}")
		logger.error(traceback.format_exc())

		clean_up_audio(audio_input)

		# Build minimal common metadata if available
		common_meta = {
			"conversation_id": locals().get("conversation_id", ""),
			"conversation_chunk_id": locals().get("conversation_chunk_id", ""),
			"metadata_str": locals().get("metadata_str", ""),
			"enable_timestamps": locals().get("enable_timestamps", False),
			"language": locals().get("job_input_language", None),
		}

		# Return a generic error payload to the caller (includes metadata)
		return {
			**common_meta,
			"error": str(e),
			"message": "An unhandled error occurred while processing the request.",
		}


runpod.serverless.start({"handler": handler})
