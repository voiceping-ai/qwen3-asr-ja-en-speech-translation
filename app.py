"""
Gradio demo for Qwen3-ASR JA-EN Speech Translation.

Bidirectional speech translation between Japanese and English
using a fine-tuned Qwen3-ASR-1.7B model. Runs on CPU.

- EN audio -> JA text
- JA audio -> EN text
"""

import gradio as gr
import numpy as np
import torch
from qwen_asr import Qwen3ASRModel

MODEL_ID = "voiceping-ai/qwen3-asr-ja-en-speech-translation"

DIRECTIONS = {
    "English -> Japanese": {"language": "Japanese", "label": "EN -> JA"},
    "Japanese -> English": {"language": "English", "label": "JA -> EN"},
}

# Load model once at startup (CPU inference for HF Spaces free tier)
print("Loading Qwen3-ASR model (CPU)...")
model = Qwen3ASRModel.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
    device_map="cpu",
)
print("Model loaded.")


def translate(audio, direction):
    if audio is None:
        return "Please provide an audio input."

    cfg = DIRECTIONS[direction]

    # Gradio returns (sample_rate, numpy_array)
    sr, audio_data = audio

    # Convert to float32 and normalize if integer type
    if audio_data.dtype in (np.int16, np.int32):
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

    # Convert stereo to mono
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa

        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000

    result = model.transcribe(audio=(audio_data, sr), language=cfg["language"])
    text = result[0].text if result else ""
    return text


demo = gr.Interface(
    fn=translate,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], label="Audio Input"),
        gr.Radio(
            choices=list(DIRECTIONS.keys()),
            value="English -> Japanese",
            label="Translation Direction",
        ),
    ],
    outputs=gr.Textbox(label="Translation", lines=3),
    title="Qwen3-ASR JA-EN Speech Translation",
    description=(
        "Bidirectional speech translation between Japanese and English "
        "using a fine-tuned [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) model. "
        "Inference runs on **CPU**.\n\n"
        "- **English -> Japanese**: Speak/upload English audio to get Japanese text\n"
        "- **Japanese -> English**: Speak/upload Japanese audio to get English text\n\n"
        "Set the direction to match the **source audio language**."
    ),
    article=(
        "See the [model card](https://huggingface.co/voiceping-ai/"
        "qwen3-asr-ja-en-speech-translation) for more details."
    ),
    examples=[
        ["example_en.wav", "English -> Japanese"],
        ["example_ja.wav", "Japanese -> English"],
    ],
    flagging_mode="never",
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(show_error=True)
