# test the whisperx model

import whisperx

model = whisperx.load_model("small", device="cpu", compute_type="int8")

audio = whisperx.load_audio("audio_Arthur.wav")

result = model.transcribe(audio, 
batch_size=1, language="en", print_progress=True)
# model_a, metadata = whisperx.load_align_model(language_code="en", device="cpu")
# # result = whisperx.align(result["segments"], model_a, metadata, audio, "cpu")
print(result["segments"])