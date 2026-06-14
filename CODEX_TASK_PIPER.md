# Codex task: replace gTTS realtime speech synthesis with Piper

Repository: trinitron88/ChinTranslator
Base branch: main
Issue: #11

Goal
Replace per-utterance gTTS MP3 generation in the realtime Hugging Face Space with local Piper TTS to reduce latency, network dependency, and phone battery drain.

Context
hf_space/app.py currently calls gTTS, writes an MP3 temp file, reloads it via librosa, converts it to PCM, and yields it back to FastRTC. This works but is slow and network-bound. The project roadmap already calls out Piper as the next step for low-latency local TTS.

Suggested implementation
- Work primarily in hf_space/app.py and hf_space/requirements.txt.
- Add backend config with environment variables such as TTS_BACKEND, PIPER_MODEL, and PIPER_CONFIG.
- Keep gTTS as fallback while Piper is being tested.
- Implement a speak_piper(text, lang) path for English output in Chin to English mode.
- Return FastRTC-compatible sample rate and int16 PCM audio.
- Avoid shelling out per phrase if a Python API or persistent process is practical; otherwise document the tradeoff.
- English to Chin can remain text-only unless a suitable Chin voice exists.
- Update REALTIME.md and/or README.md with required Space variables and model setup.

Acceptance criteria
- Chin to English mode speaks English using Piper when configured.
- App still boots if Piper is unavailable and falls back cleanly to gTTS or text-only.
- Per-utterance latency is materially lower than gTTS in local or HF testing.
- No temp MP3 round-trip is required for the Piper path.
- Docs explain how to configure Piper in the Space.

Suggested validation
- Run or import hf_space/app.py enough to confirm syntax and imports.
- Test the gTTS fallback still behaves like current production.
- Test text-only mode if implemented.
- Test Piper path on a short English phrase and confirm PCM output shape and sample rate are accepted by FastRTC.
