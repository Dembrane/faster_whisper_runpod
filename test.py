# # test the whisperx model

# print("hello")

# from huggingface_hub.utils import _runtime

# _runtime._is_google_colab = False
# from faster_whisper.tokenizer import Tokenizer
# import whisperx
# from dataclasses import replace

# print("hello2")

# model = whisperx.load_model(
#     'small',
#     device="cpu",
#     compute_type="int8",
# )
# tokenizers = {}
# supported_languages = ["en", "nl", "fr"]
# for lang in supported_languages:
#     tokenizers[lang] = Tokenizer(
#         model.model.hf_tokenizer,
#         True,
#         task="transcribe",
#         language=lang,
#     )


# new_options = {
#     "initial_prompt": "Hello, how are you?",
# }

# model.options = replace(model.options, **new_options)


# audio = whisperx.load_audio("test.mp3")

# result = model.transcribe(audio, batch_size=1, language="fr", print_progress=True)

# print(result["segments"])


from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen2-0.5B", device='cpu')
messages = [
    {"role": "user", "content": "Who are you?"},
]
result = pipe(messages)
print(result[0]['generated_text'])