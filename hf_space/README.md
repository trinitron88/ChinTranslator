---
title: Hakha Chin Realtime Interpreter
emoji: 🎤
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
python_version: "3.12"
pinned: false
---

# Hakha Chin → English — real-time voice interpreter

Continuous in-ear interpreter built on [FastRTC](https://fastrtc.org): it listens,
detects when the speaker pauses, transcribes Hakha Chin with a fine-tuned
Whisper model, translates to English, and speaks the English back over WebRTC.

## Configuration (Space Settings)

- **Variable `CHIN_MODEL`** — your CT2 model repo id, e.g.
  `your-username/whisper-cnh-turbo-ct2`. `faster-whisper` downloads it on boot.
  **Required:** the Space refuses to start without it rather than silently
  serving stock Whisper (which knows no Hakha Chin and emits garbage). Set
  `ALLOW_STOCK=1` only to intentionally demo the base model.
- **Secret `HF_TOKEN`** — used to fetch Cloudflare TURN credentials (and to pull
  the model if its repo is private).
- **Hardware** — needs a **GPU** tier for real-time; CPU is too slow for
  `large-v3-turbo`.
- **Variable `DENOISE`** *(optional)* — ambient-noise suppression on the 16 kHz
  signal before ASR: `noisereduce` (default, light spectral gating), `df`
  (DeepFilterNet — neural, cleaner; also add `deepfilternet` to
  `requirements.txt`), or `off`. `DENOISE_AMOUNT` (0–1, default 0.85) eases the
  reduction; `DENOISE_STATIONARY=0` switches to the slower non-stationary mode
  for fluctuating (non-steady) noise.

## Publishing the model (one-time)

The Space runs on HF infra and **cannot read a Drive path** — `CHIN_MODEL` must
be an HF *model-repo id*. After `export_model.py` produces the CT2 folder, push
it to a model repo and point the Space at it:

```bash
python hf_space/upload_model.py --repo your-username/whisper-cnh-turbo-ct2
# then set the Space Variable CHIN_MODEL to that repo id and restart
```

## Notes

- Latency of ~2–5s behind the speaker is normal for live interpretation.
- TTS is gTTS (placeholder); swap to Piper for lower latency.
- Companion to the batch translator and `realtime.py` in the source repo.
