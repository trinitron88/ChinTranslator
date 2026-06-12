# Real-time in-ear interpreter (v2 prototype)

`realtime.py` is the streaming version of the translator: instead of
record → stop → translate, it listens continuously and speaks English into your
ear a few seconds behind the speaker.

```
phone browser (mic in, earbud out)
    ⇅  live audio stream (WebRTC, via FastRTC)
GPU backend (Colab / cloud):
    VAD → Chin Whisper → Google translate → TTS → stream English back
```

The phone is just a microphone and a speaker; all the model work stays on the
GPU. This reuses the same fine-tuned model and `cnh`→EN translation as the batch
app on `main`.

## Run it (on a GPU)

```python
# In Colab, with Drive mounted and the repo cloned to /content/CT:
!cd /content/CT && CHIN_MODEL=/content/drive/MyDrive/ChinTranslator/model_v5/whisper-cnh-turbo-ct2 \
    python realtime.py
```

Open the printed share link **on your phone**, allow microphone access, put a
Bluetooth earbud in, and start talking (or point it at a Chin speaker).

## Status: prototype — expect to iterate

This is a working skeleton, not a finished product. Known things to expect/fix:

1. **WebRTC connectivity (most likely first hurdle).** A plain share link often
   only connects on the same LAN. For a phone on cellular talking to a Colab
   backend, WebRTC needs a **TURN relay**. Two options, in order of preference:
   - **Static TURN creds** — set `TURN_URLS` (comma-separated `turn:` URLs),
     `TURN_USERNAME`, and `TURN_CREDENTIAL` (e.g. from a free ExpressTURN or
     Metered account). This is what the HF Space uses; no broker fetch.
   - **HF/Cloudflare broker** — set `HF_TOKEN` and FastRTC fetches free
     Cloudflare TURN credentials. This fetch has been flaky (DNS failures);
     if the phone can't connect, this is why.

2. **Latency ~2–5s** behind the speaker is normal — even human interpreters lag.
   You won't get instant.

3. **TTS is gTTS for now**, which round-trips to Google per phrase and is too
   slow for snappy real-time. Next step: swap in **Piper** (local neural TTS,
   sub-second, runs on CPU/GPU) in `speak_en()`.

4. **You'll hear English over their ongoing Chin** — the interpreter's
   split-attention effect. Workable, but real.

5. **Overlapping voices still break it.** Point the phone at the one person you
   want to hear; it favors the loudest/nearest voice.

6. **FastRTC's API moves fast.** If an import or call signature in `realtime.py`
   is off, `pip show fastrtc` and adjust against the installed version.

7. **Colab is fine for a prototype, not for an always-on service** — it
   disconnects after ~90 min idle and is ephemeral. A stable "earbud in all day"
   experience eventually wants a small cloud GPU, or the on-device path below.

## Roadmap

- **v2 (this):** FastRTC streaming on a GPU backend, phone browser + earbud.
- **v2.1:** Piper TTS for low latency; tune VAD/chunking; partial results.
- **v3:** on-device — the fine-tuned model converted to run locally
  (whisper.cpp + Piper), no server, works offline. The version that becomes a
  thing you own rather than a thing you launch.
