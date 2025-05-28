# test the whisperx model

print("hello")

from huggingface_hub.utils import _runtime

_runtime._is_google_colab = False

import whisperx

print("hello2")

# output = transcribe_task(
#   args={
#     # "model_name": "deepdml/faster-whisper-large-v3-turbo-ct2",
#     "model": "small",
#     "model_dir": "models",
#     "device": "cpu",
#     "compute_type": "int8",
#     "batch_size": 1,
#     "language": "en",
#     "initial_prompt": "Hello, how are you?",
#     "output_format": "json",
#     "output_dir": "output",
#     "print_progress": True,
#     "verbose": True,
#     "task": "transcribe",
#     "no_align": True,
#     "device_index": 0,
#     "diarize": False,
#   },
#   parser=None
# )

# print(output)

# model = whisperx.load_model("deepdml/faster-whisper-large-v3-turbo-ct2", device="cpu", compute_type="int8")

model = whisperx.load_model(
    # "small",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    device="cpu",
    compute_type="int8",
    language="en",
    download_root="models",
    local_files_only=True,
)

audio = whisperx.load_audio("test.mp3")

result = model.transcribe(audio, batch_size=1, language="en", print_progress=True)

print(result["segments"])
