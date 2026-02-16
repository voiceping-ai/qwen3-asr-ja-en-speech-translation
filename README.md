---
title: Qwen3-ASR JA-EN Speech Translation
emoji: "\U0001F30F"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: apache-2.0
language:
  - en
  - ja
tags:
  - qwen3-asr
  - speech-translation
  - japanese
  - english
pipeline_tag: automatic-speech-recognition
---

# Qwen3-ASR JA-EN Speech Translation

Bidirectional speech translation between Japanese and English using a fine-tuned [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) model. Inference runs on **CPU**.

| Direction | Input | Output |
|-----------|-------|--------|
| EN -> JA | English audio | Japanese text |
| JA -> EN | Japanese audio | English text |

## Model Details

### Architecture

| Component | Details |
|-----------|---------|
| Base model | [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |
| Architecture | Qwen3-ASR (audio encoder + language model) |
| Total parameters | ~1.7B |
| Max audio length | 30 seconds |

### Training

Fine-tuned for bidirectional speech translation (EN<->JA) using ~1.27M paired audio-text translation samples across 8 datasets.

**Training methodology:**

- **Task**: Supervised fine-tuning (SFT) for speech translation
- **Optimizer**: AdamW
- **Learning rate**: 1e-5 with cosine scheduler
- **Epochs**: ~1.3 (best checkpoint at epoch 1.16)
- **Effective batch size**: 64 (batch_size=8, grad_acc=8)
- **Warmup**: 3% of total steps
- **Audio filtering**: 0.5-30 seconds duration
- **Training data**: ~1.27M samples (bidirectional EN<->JA)

### How Translation Direction Works

The translation direction is controlled via the `language` parameter in the Qwen3-ASR `transcribe()` call:

- `language="Japanese"` = EN audio -> **JA text**
- `language="English"` = JA audio -> **EN text**

The `language` parameter specifies the **target output language**.

### Evaluation

Evaluated on the [FLEURS](https://huggingface.co/datasets/google/fleurs) test set for both translation directions.

## Usage

### Installation

```bash
pip install torch transformers>=4.57.0 qwen3-asr librosa
```

### EN audio -> JA text

```python
import torch
import librosa
from qwen_asr import Qwen3ASRModel

MODEL_ID = "voiceping-ai/qwen3-asr-ja-en-speech-translation"

model = Qwen3ASRModel.from_pretrained(MODEL_ID, dtype=torch.float32, device_map="cpu")

# Load audio (16kHz mono)
audio, sr = librosa.load("english_audio.wav", sr=16000)

# EN audio -> JA text: set language to target language
result = model.transcribe(audio=(audio, sr), language="Japanese")
print(result[0].text)
```

### JA audio -> EN text

```python
# JA audio -> EN text: set language to target language
result = model.transcribe(audio=(audio, sr), language="English")
print(result[0].text)
```

### Standalone Inference Script

See [`inference.py`](inference.py) for a complete standalone script that handles audio file input, device selection, and both translation directions.

```bash
# EN audio -> JA text
python inference.py audio_en.wav --direction en2ja

# JA audio -> EN text
python inference.py audio_ja.wav --direction ja2en

# Use GPU
python inference.py audio.wav --direction en2ja --device cuda:0
```

## Example Predictions

Predictions on [FLEURS](https://huggingface.co/datasets/google/fleurs) test set samples.

### EN -> JA

| Source (EN audio) | Prediction (JA text) |
|---|---|
| however due to the slow communication channels styles in the west could lag behind by 25 to 30 year | しかし、通信の遅さのために、西洋のスタイルは25年から30年も遅れてしまう可能性があります。 |
| the harbor was the site of an infamous naval standoff in 1889 when seven ships from germany the us and britain refused to leave the harbor | この港は、1889年にドイツ、アメリカ、イギリスから7隻の船が港を離れることを拒否したことで、悪名高い海軍の対立の場となった場所です。 |
| plants make oxygen which humans breathe and they take in carbon-dioxide which humans exhale that is breathe out | 植物は酸素を生成し、人間が呼吸するものです。また、植物は二酸化炭素を吸収し、人間が排出するものです。つまり、植物は人間の呼吸を助けるのです。 |
| according to japan's nuclear agency radioactive caesium and iodine has been identified at the plant | 日本原子力規制委員会によると、その工場で放射性カシウムやヨウ素が検出されたとのことです。 |
| appropriate use of blogs can empower students to become more analytical and critical | ブログの適切な利用によって、学生はより分析的で批判的な思考を身につけることができます。 |

### JA -> EN

| Source (JA audio) | Prediction (EN text) |
|---|---|
| インターネットで 敵対的環境コース について検索すると おそらく現地企業の住所が出てくるでしょう | If you search for "anti-environmental courses" on the internet, you'll probably find the addresses of local companies that offer such courses. |
| スブリングボックにとっては 5連敗が止まりました | For the Spring Box, this is the fifth consecutive win. |
| 実際には 洞窟の存在を知っていても見つけるのは容易ではありません 洞窟に入ると そこは完全に隔離された空間です | In reality, even if one knows about the existence of caves, it's not easy to find them. Once you enter a cave, it becomes a completely isolated space. |
| 1940年8月15日 連合国は南フランスに侵攻し その侵攻は ドラグーン作戦 と呼ばれました | On August 15, 1940, the Allied forces invaded southern France. This invasion was known as the "Draconian Campaign". |
| プリトヴィッツェ湖群国立公園は アルプスと地中海の植生が混在する広大な森主にブナ トウヒ モミに大小16の湖が点在し 絶景で溢れる世界遺産です | Pitbitte National Park is a highland forest where both alpine and Mediterranean vegetation coexist. It's home to large lakes, mainly located in areas covered with birch, pine, and poplar trees. It's a World Heritage Site, known for its stunning landscapes. |

## Limitations

- **Audio length**: Best performance on audio segments under 30 seconds
- **Language pair**: Only supports EN<->JA translation (not general-purpose multilingual)
- **Domain**: Trained primarily on general-domain speech; specialized domains (medical, legal, etc.) may have lower accuracy

## License

Apache 2.0
