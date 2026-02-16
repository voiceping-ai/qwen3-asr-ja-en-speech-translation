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

Side-by-side comparison on [FLEURS](https://huggingface.co/datasets/google/fleurs) test set samples across four models:

Quality scores rated on FLEURS test samples (1-5 scale: accuracy + fluency, scored by Claude).

| Model | Parameters | EN->JA | JA->EN |
|-------|-----------|--------|--------|
| [OpenAI Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) | 1.55B | N/A (English-only output) | 3.2/5 |
| [Meta SeamlessM4T v2 Large](https://huggingface.co/facebook/seamless-m4t-v2-large) | 1.50B | 3.8/5 | 3.0/5 |
| [Whisper JA-EN (ours)](https://huggingface.co/voiceping-ai/whisper-ja-en-speech-translation) | 756M | 2.6/5 | 2.4/5 |
| **Qwen3-ASR JA-EN (ours)** | 1.7B | **4.2/5** | **4.0/5** |

### EN -> JA

| Source (EN audio) | [OpenAI Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) | [Meta SeamlessM4T v2](https://huggingface.co/facebook/seamless-m4t-v2-large) | [Whisper JA-EN (ours)](https://huggingface.co/voiceping-ai/whisper-ja-en-speech-translation) | **Qwen3-ASR JA-EN (ours)** |
|---|---|---|---|---|
| through the night between 150 and 200 copies were made now known as dunlap broadsides | N/A | 今ではダンラップ・ブロードサイズとして知られる 150 枚から 200 枚のコピーが作られました | 百五十から二百までのコピーが生成されました。 | 150から200本のコピーが作られました。これらは今では「Dunlap Broadside」として知られています。 |
| the find also grants insight into the evolution of feathers in birds | N/A | この発見は羽の進化と鳥の進化についても洞察を深めます | この発見は、鳥の羽の形にも影響を与えます。 | この発見は、羽や鳥の進化についても洞察を与えてくれます。 |
| we don't know for sure but it may have had a forked tongue its diet included turtles large fish other mosasaurs and it may even have been a cannibal | N/A | 確かには分かりませんが舌が分岐していたかもしれませんその食事にはカメ大拳他のモササウルスなどが含まれていたそして人間食人だったかもしれません | 確かはわかりませんが、おそらくフォークタンというものでしょう。そのダイには、カメや大きな五頭など、他のモーサーも含まれていました。また、彼は大麻でもありました。 | 確かなことはわかりませんが、おそらくその動物は舌が二本だったかもしれません。その動物の食事内容には、カメや大きな魚、他のモササウルスなどが含まれていました。また、その動物はおそらく人を食べていたのかもしれません。 |
| australia's mitchell gourley finished eleventh in the men's standing super-g czech competitor oldrich jelinek finished sixteenth in the men's sitting super-g | N/A | オーストラリアのミッチェル・ゴーリーは男子スタンドGで11位,チェコのオルドリッチ・デレニックは男子スタンドGで16位となりました. | オーストラリアのミッチェルゴーリーは、男子スタンディングスーパーGで十一位を獲得しました。一方、チェコの競争相手であるオルドリッチデレニックは、男子スタンディングスーパーGで十六位を獲得しました。 | オーストラリアのミッチェル・ゴーリーは、男子立位のスーパージューコースで11位を獲得しました。チェコの競技者であるオルドリッチ・ドレニックは、男子座位のスーパージューコースで16位を獲得しました。 |
| in many other cities of italy and in the rest of the world particularly in poland similar setups were made which were viewed by a great number of people | N/A | イタリアの他の多くの都市や世界中の他の都市,特にポーランドでは,多くの人々が見た同様の設定が作られました. | イタリアやその他の国々でも、特にポーランドでは、同様の設備が設置されていました。これらの設備は、多くの人々によって見られました。 | イタリアの他の多くの都市や、世界の他の地域、特にポーランドでも、同様のシステムが設置されていました。そして、それらは多くの人々によって見られました。 |

### JA -> EN

| Source (JA audio) | [OpenAI Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) | [Meta SeamlessM4T v2](https://huggingface.co/facebook/seamless-m4t-v2-large) | [Whisper JA-EN (ours)](https://huggingface.co/voiceping-ai/whisper-ja-en-speech-translation) | **Qwen3-ASR JA-EN (ours)** |
|---|---|---|---|---|
| 残念ながら 運転手の行動を100%の確率で予測することはできないので 交通の流れを研究することは困難です | Unfortunately, it is difficult to predict the behavior of the driver with 100% probability, so it is difficult to study the flow of traffic. | Unfortunately, we can't predict the behavior of drivers with a hundred percent probability, so it's difficult to study the traffic flow. | Unfortunately, it's impossible to predict the behavior of drivers with 100% certainty. Therefore, it's difficult to study the flow of traffic. | Unfortunately, it's not possible to predict the actions of drivers with 100% certainty. Therefore, it's difficult to study the flow of traffic. |
| しかし シェンゲン圏は この点では一国のように機能します | However, the declaration key works like a single click. | But the declaration of rights works like a miracle in this case. | However, in this regard, the pericard functions effectively. | However, the right to declare war functions in this case like a single country. |
| 夜空の下でピラミッドに映像が浮かび ピラミッドが次々とライトアップされます | The image of the pyramid is reflected under the night sky, and the pyramid is lighted up one after another. | Under the night sky, the pyramids are illuminated, and the pyramids are lit up one by one. | The pyramid is lit up one after another. | Under the night sky, images of pyramids are illuminated. The pyramids are lit up one after another. |
| バーチャルチームは 従来のチームと同じ水準の卓越性が求められますが 微妙な違いがあります | Virtual teams require the same level of flexibility as conventional teams, but there is a subtle difference. | A virtual team requires the same level of tact as a traditional team, but there's a subtle difference. | The virtual team requires a level of durability, just like the traditional team. However, there are some slight differences between them. | Virtual teams require the same level of collaboration as traditional teams. However, there are some subtle differences between them. |
| キルヒネル夫人は アルゼンチン劇場で大統領選に出馬するという意向を表明しました 2005年にブエノスアイレス州の代表団の一員として上院選への出馬を表明したのもこの劇場でした | Mrs. Kirchner expressed her intention to appear in the presidential election at the Argentine theater. In 2005, she was one of the representatives of Buenos Aires, and she expressed her intention to appear in the presidential election at this theater. | Mrs. Kilphinel expressed her intention to run for president in the Argentine theater, and in 2005 she was one of the delegates from the state of Buenos Aires to run for the Senate. | Kylhine announced her intention to run for president in the Argentina theater. In 2005, she also announced her participation in the top-ranked team of the Ueno-Styles state as a representative of the Ueno-Stylus. | Mrs. Kirchner announced her intention to run for president at the Alzen Theatre. In 2005, she also announced her candidacy for the Senate as part of the delegation from Buenos Aires. It was at this theatre that she made her announcement. |

## Limitations

- **Audio length**: Best performance on audio segments under 30 seconds
- **Language pair**: Only supports EN<->JA translation (not general-purpose multilingual)
- **Domain**: Trained primarily on general-domain speech; specialized domains (medical, legal, etc.) may have lower accuracy

## License

Apache 2.0
