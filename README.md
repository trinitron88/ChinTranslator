# Hakha Chin Speech-to-Text Translator

A fine-tuned Whisper model for transcribing Hakha Chin (`cnh`) speech and
translating it to English. Built to help bridge language barriers in Hakha
Chin-speaking communities.

## 🎯 Overview

**Current status: V6** — a LoRA fine-tune of `openai/whisper-large-v3-turbo`
trained on the [Common Voice](https://commonvoice.mozilla.org/) Hakha Chin
dataset (community-recorded, pre-aligned utterances with validated
transcripts). Earlier versions (V1–V4) trained on Bible audio; that data and
its alignment pipeline are retired to `archive/` — Common Voice gives cleaner
alignment, more speakers, and conversational vocabulary.

V6 is an accuracy push for **conversational** speech: a fixed surrogate
language token (matching train and inference prompts), all of train+dev plus
mined unvalidated clips as training data, waveform + SpecAugment augmentation,
broader LoRA, best-checkpoint selection, and WER measured on the held-out
official test split (158 unseen speakers). Rationale and results plan:
[V6_PLAN.md](V6_PLAN.md).

There are three ways to use the model:

| App | What it does |
|-----|--------------|
| `gradio_interface.py` | Batch: upload/record audio → Chin transcript + English translation + spoken English |
| `realtime.py` | Streaming prototype: phone mic → GPU backend → English in your earbud a few seconds behind the speaker (see [REALTIME.md](REALTIME.md)) |
| `hf_space/` | The realtime app packaged for Hugging Face Spaces (WebRTC + TURN work there; Colab can't carry WebRTC media) |

Translation is Google Translate's endpoint called directly with the source
pinned to `cnh` (deep-translator's language list lacks Hakha Chin, and
autodetect misreads it). TTS is gTTS.

## 🚀 Quick start

```bash
git clone https://github.com/trinitron88/ChinTranslator.git
cd ChinTranslator

# Serve the batch app (downloads stock large-v3 if CHIN_MODEL is unset)
python gradio_interface.py

# Serve the fine-tuned model (after training + export, see below)
CHIN_MODEL=whisper-cnh-turbo-ct2 python gradio_interface.py
```

Scripts self-install their Python dependencies on first run (they're built to
be `!python`-run from Colab cells). `gradio_interface.py` also needs **ffmpeg**
on the PATH (`apt install ffmpeg` / `brew install ffmpeg`).

## 🔧 Training pipeline (V6)

Designed for a free Colab T4 (16 GB). The base model is frozen and loaded in
8-bit; only small LoRA adapters train — minutes per epoch, and it resists
overfitting on a few hours of audio.

```bash
python prepare_data.py     # fetch Common Voice cnh → data/cv_cnh/ (train/val/test)
python train.py            # LoRA fine-tune → whisper-cnh-turbo-lora/ (adapter)
python export_model.py     # merge adapter + convert → whisper-cnh-turbo-ct2/ (CTranslate2)
python evaluate_model.py --model whisper-cnh-turbo-ct2   # WER/CER on held-out test

# data flywheel: mine the 3.2h unvalidated pool with the model you just trained
python mine_other.py --model whisper-cnh-turbo-ct2
python prepare_data.py --other-keep other_keep.json      # then retrain
```

On Colab with Drive mounted, `train.py`/`export_model.py` default their outputs
into `/content/drive/MyDrive/ChinTranslator/` so a runtime reset doesn't eat
the model. Explicit `--out`/`--adapter` flags always win.

The export step exists because the serving apps use
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2),
which understands neither PEFT adapters nor raw HF checkpoints: the adapter is
merged into full-precision base weights, then converted to CT2 format.

## 📁 Project structure

```
.
├── prepare_data.py        # Common Voice cnh → data/cv_cnh/ (train/val/test)
├── train.py               # LoRA + 8-bit fine-tune of whisper-large-v3-turbo
├── export_model.py        # merge adapter → convert to CTranslate2
├── evaluate_model.py      # WER/CER on the held-out official test split
├── mine_other.py          # pseudo-label mining of the unvalidated CV pool
├── gradio_interface.py    # batch web app (upload/record → transcript + translation)
├── realtime.py            # streaming in-ear interpreter prototype (FastRTC)
├── REALTIME.md            # realtime architecture, setup, roadmap
├── hf_space/              # Hugging Face Space (realtime app + deploy script)
│   ├── app.py             #   Spaces entrypoint (direction toggle, mic sensitivity, transcript)
│   └── deploy_colab.py    #   push hf_space/ to the Space from Colab
├── ChinTranslator_V5_Colab.ipynb   # one-stop Colab notebook for the pipeline
└── archive/               # retired Bible-data pipeline (V1–V4) + superseded scripts
```

## 🛠️ Technical details

- **Base model**: `openai/whisper-large-v3-turbo` (0.8B), frozen, 8-bit
- **Adapter**: LoRA r=32, α=64, dropout 0.05 on all attention projections +
  MLP (`--lora-targets attn` restores the V5 q/v-only adapter)
- **Language token**: Whisper has no `cnh` token, so training fixes a
  surrogate (`id` — where Whisper's detector already puts cnh speech) and
  serving forces the same one via `chin_metadata.json`, keeping the decoder
  prompt identical between training and inference
- **Data**: Common Voice 17 `cnh` — all of train+dev, plus vote-filtered /
  pseudo-label-mined clips from the unvalidated `other` pool; small val
  carve-out for early stopping; the official test split is never trained on
  and serves as the WER benchmark (best published: 31.4% WER, wav2vec2-xlsr)
- **Augmentation**: per-epoch waveform augmentation (gain, noise, speed) +
  SpecAugment, train split only
- **Selection**: best checkpoint by eval loss + early stopping
- **Serving**: faster-whisper / CTranslate2, float16 on GPU, int8 on CPU

## 🌐 Hugging Face Space (realtime)

The Space (`bsantisi/chin-realtime`) serves the streaming interpreter with a
Chin↔English direction toggle and a mic-sensitivity slider (helps AirPods /
quiet Bluetooth mics). Configuration via Space settings:

- **Variable `CHIN_MODEL`** — HF repo id of the uploaded CT2 model
- **Secret `HF_TOKEN`** — model download + Cloudflare TURN broker fallback
- **Secrets `TURN_URLS` / `TURN_USERNAME` / `TURN_CREDENTIAL`** — preferred
  static TURN relay (e.g. a free ExpressTURN/Metered account); the broker
  fetch is unreliable

Deploy from Colab with `hf_space/deploy_colab.py`.

## 🔄 Model versions

| Version | Data | Status | Notes |
|---------|------|--------|-------|
| V1–V3 | Bible audio (Mark/Matthew) | ❌ retired | alignment pipeline, repetition/alignment failures |
| V4 | Bible audio, 1,375 segments | ❌ superseded | worked, but male read-speech, biblical domain only |
| V5 | Common Voice `cnh` train+dev (80%) | ❌ superseded | LoRA on large-v3-turbo, no language token, no held-out eval |
| **V6** | CV train+dev+mined `other` | ✅ **current** | surrogate lang token, augmentation, broader LoRA, WER-benchmarked ([V6_PLAN.md](V6_PLAN.md)) |

## 🚀 Roadmap

Data (the big lever — see the backlog in [V6_PLAN.md](V6_PLAN.md)):
- CMU Wilderness `CNHBSM`: 21.7 h of aligned cnh speech (10× current data)
- Synthetic training pairs via Meta's `facebook/mms-tts-cnh` TTS voice
- Newer Common Voice releases (Mozilla Data Collective; ~6 h recorded in v24)
- Field-recording flywheel: record → transcribe → native-speaker correction → retrain

Engineering:
- Piper TTS in the realtime path (gTTS round-trips to Google per phrase)
- Partial/streaming results and VAD tuning for lower latency
- On-device (whisper.cpp + Piper) — offline, no server; MADLAD-400 for offline cnh→en translation
- Ensemble/cross-check with Meta MMS ASR (supports cnh) for confidence scoring

## 📝 License

For educational and language-preservation purposes. Please respect the
licenses of OpenAI Whisper (Apache 2.0), Mozilla Common Voice (CC-0), and the
Transformers ecosystem (Apache 2.0).

## 📧 Contact

[GitHub Issues](https://github.com/trinitron88/ChinTranslator/issues)
