"""
Standalone inference script for qwen3-asr-ja-en-speech-translation.

Translates audio files between English and Japanese using a fine-tuned Qwen3-ASR model.

Usage:
    python inference.py <audio_file> --direction <en2ja|ja2en> [--device DEVICE]

Examples:
    python inference.py audio_en.wav --direction en2ja
    python inference.py audio_ja.wav --direction ja2en
    python inference.py audio.wav --direction en2ja --device cuda:0
"""

import argparse

import librosa
import torch
from qwen_asr import Qwen3ASRModel

MODEL_ID = "voiceping-ai/qwen3-asr-ja-en-speech-translation"

DIRECTIONS = {
    "en2ja": {"language": "Japanese", "label": "EN -> JA"},
    "ja2en": {"language": "English", "label": "JA -> EN"},
}


def translate(audio_path: str, direction: str, device: str, model_id: str):
    cfg = DIRECTIONS[direction]

    print(f"Model     : {model_id}")
    print(f"Direction : {cfg['label']}")
    print(f"Audio     : {audio_path}")
    print(f"Device    : {device}")

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = Qwen3ASRModel.from_pretrained(model_id, dtype=dtype, device_map=device)

    audio, sr = librosa.load(audio_path, sr=16000)
    print(f"Duration  : {len(audio) / sr:.1f}s")

    result = model.transcribe(audio=(audio, sr), language=cfg["language"])
    text = result[0].text if result else ""
    print(f"\nResult    : {text}")
    return text


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR JA-EN Speech Translation")
    parser.add_argument("audio", help="Path to audio file (16kHz mono recommended)")
    parser.add_argument(
        "--direction",
        required=True,
        choices=list(DIRECTIONS.keys()),
        help="Translation direction: en2ja or ja2en",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda:0 if available)",
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help=f"HuggingFace model ID (default: {MODEL_ID})",
    )
    args = parser.parse_args()

    translate(args.audio, args.direction, args.device, args.model_id)


if __name__ == "__main__":
    main()
