#!/usr/bin/env python3
"""
app.py — Hugging Face Spaces entrypoint for the real-time Hakha Chin → English
interpreter (FastRTC streaming).

Deployed to Spaces because Colab can't carry WebRTC media. Here, WebRTC + TURN
work. The model is loaded from an HF model repo by id — set the Space VARIABLE
`CHIN_MODEL` to your uploaded CT2 repo (e.g. "your-username/whisper-cnh-turbo-ct2"),
and the Space SECRET `HF_TOKEN` so FastRTC can fetch Cloudflare TURN credentials
(and so a private model repo can be downloaded).

Dependencies are installed from requirements.txt at build time.
"""

import os
import json
import tempfile

import numpy as np
import librosa
import torch
from faster_whisper import WhisperModel
from fastrtc import Stream, ReplyOnPause

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Set CHIN_MODEL to your CT2 model repo id; faster-whisper downloads it from HF.
MODEL_NAME = os.environ.get("CHIN_MODEL", "openai/whisper-large-v3")
print(f"Loading model: {MODEL_NAME} on {DEVICE}")
MODEL = WhisperModel(MODEL_NAME, device=DEVICE,
                     compute_type="float16" if DEVICE == "cuda" else "int8")
print("✓ Model loaded.")

CHIN_CODE = "cnh"


def to_en(text_chin: str) -> str:
    text = (text_chin or "").strip()
    if not text:
        return ""
    import urllib.parse
    import urllib.request
    url = ("https://translate.googleapis.com/translate_a/single"
           f"?client=gtx&sl={CHIN_CODE}&tl=en&dt=t&q=" + urllib.parse.quote(text))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
        return "".join(seg[0] for seg in data[0] if seg and seg[0]).strip() or text
    except Exception as e:
        print(f"[translate] failed ({e})")
        return text


def speak_en(text_en: str):
    text = (text_en or "").strip()
    if not text:
        return None
    try:
        from gtts import gTTS
        mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        gTTS(text=text, lang="en").save(mp3)
        audio, sr = librosa.load(mp3, sr=24000, mono=True)
        os.unlink(mp3)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).reshape(1, -1)
        return sr, pcm
    except Exception as e:
        print(f"[tts] failed ({e})")
        return None


def on_utterance(audio):
    sr, samples = audio
    samples = np.asarray(samples).astype(np.float32).flatten()
    peak = np.max(np.abs(samples)) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / 32768.0
    if sr != 16000:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
    segs, _ = MODEL.transcribe(samples, task="transcribe", beam_size=5, vad_filter=False)
    chin = "".join(s.text for s in segs).strip()
    if not chin:
        return
    english = to_en(chin)
    print(f"CHIN: {chin!r}  →  EN: {english!r}", flush=True)
    out = speak_en(english)
    if out is not None:
        yield out


# TURN for Spaces. Per FastRTC's docs, pass the ASYNC Cloudflare credential
# function (HF brokers ~10 GB/mo free TURN via your HF token). It's invoked
# per-connection, so it satisfies the Spaces startup check WITHOUT a credential
# fetch at import time — the sync fetch was DNS-failing during startup.
from fastrtc import get_cloudflare_turn_credentials_async

_HF = os.environ.get("HF_TOKEN")


async def _turn_credentials():
    return await get_cloudflare_turn_credentials_async(hf_token=_HF)


stream = Stream(
    handler=ReplyOnPause(on_utterance),
    modality="audio",
    mode="send-receive",
    rtc_configuration=_turn_credentials,
)

# Spaces (gradio SDK) serves this `demo` object.
demo = stream.ui

if __name__ == "__main__":
    demo.launch()
