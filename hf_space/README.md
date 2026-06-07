---
title: Hakha Chin Realtime Interpreter
emoji: 🎤
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Hakha Chin → English — real-time voice interpreter

Continuous in-ear interpreter built on [FastRTC](https://fastrtc.org): it listens,
detects when the speaker pauses, transcribes Hakha Chin with a fine-tuned
Whisper model, translates to English, and speaks the English back over WebRTC.

## Configuration (Space Settings)

- **Variable `CHIN_MODEL`** — your CT2 model repo id, e.g.
  `your-username/whisper-cnh-turbo-ct2`. `faster-whisper` downloads it on boot.
- **Secret `HF_TOKEN`** — used to fetch Cloudflare TURN credentials (and to pull
  the model if its repo is private).
- **Hardware** — needs a **GPU** tier for real-time; CPU is too slow for
  `large-v3-turbo`.

## Notes

- Latency of ~2–5s behind the speaker is normal for live interpretation.
- TTS is gTTS (placeholder); swap to Piper for lower latency.
- Companion to the batch translator and `realtime.py` in the source repo.
