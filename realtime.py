#!/usr/bin/env python3
"""
realtime.py — continuous, in-ear Hakha Chin → English interpreter (v2 prototype).

Architecture:

    phone browser (mic in, earbud out)
        ⇅  live audio stream (WebRTC, via FastRTC)
    GPU backend (Colab / cloud):
        VAD → Chin Whisper → Google translate → TTS → stream English back

You wear a Bluetooth earbud, the phone streams what it hears, and a few seconds
later the English comes back into your ear. Run this on a GPU, open the share
link on your phone, put the earbud in.

    CHIN_MODEL=/content/drive/MyDrive/ChinTranslator/model_v5/whisper-cnh-turbo-ct2 \
        python realtime.py

Honest status — this is a PROTOTYPE skeleton, expect to iterate on first run:
  * Latency of ~2–5s behind the speaker is normal for live interpretation.
  * WebRTC across networks (phone on cellular ↔ Colab) needs a TURN relay; a
    plain share link often only works on the same LAN. See REALTIME.md.
  * gTTS is used here for simplicity but round-trips to Google per phrase — too
    slow for snappy real-time. Swapping in Piper (local, fast) is the next step.
  * FastRTC's API/version moves fast; if an import or signature below is off,
    check `pip show fastrtc` and the FastRTC docs, then adjust.

This file is intentionally self-contained (it does NOT import gradio_interface,
which would launch the batch app on import). Shared logic — model load + the
cnh→EN translate — is duplicated here on purpose; a future cleanup could pull it
into a shared core module used by both apps.
"""

import os
import sys
import json
import tempfile
import subprocess


def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)


_pip("fastrtc", "faster-whisper", "gTTS", "soundfile", "librosa", "numpy", "torch")

import numpy as np  # noqa: E402
import librosa  # noqa: E402
import torch  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from fastrtc import Stream, ReplyOnPause  # noqa: E402

# ---------------- Model ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Stock fallback matches the trained base (large-v3-turbo) but is NOT fine-tuned
# — it knows no Hakha Chin. Set CHIN_MODEL to your CT2 model in real use.
STOCK_FALLBACK = "large-v3-turbo"
MODEL_NAME = os.environ.get("CHIN_MODEL", STOCK_FALLBACK)
print("=" * 64)
print(f"  Real-time Chin→EN  |  MODEL: {MODEL_NAME}  |  DEVICE: {DEVICE}")
if MODEL_NAME == STOCK_FALLBACK:
    print(f"  ⚠️  stock {STOCK_FALLBACK} (NOT fine-tuned, knows no Hakha Chin). "
          f"Set CHIN_MODEL=.../whisper-cnh-turbo-ct2")
print("=" * 64)
MODEL = WhisperModel(MODEL_NAME, device=DEVICE,
                     compute_type="float16" if DEVICE == "cuda" else "int8")
print("✓ Model loaded.")

CHIN_CODE = "cnh"


def _chin_lang_token(model_ref):
    """Surrogate language token the model was trained with (V6+ adapters; see
    train.py). Forcing the training-time token at inference beats auto-detect,
    which flaps between id/km/ms per utterance. CHIN_LANG env overrides
    (CHIN_LANG="" → auto-detect). None for V5-era models → old behavior."""
    if "CHIN_LANG" in os.environ:
        return os.environ["CHIN_LANG"] or None
    from pathlib import Path
    meta = Path(model_ref) / "chin_metadata.json"
    if meta.is_file():
        try:
            return json.loads(meta.read_text(encoding="utf-8")).get("language_token")
        except Exception as e:  # noqa: BLE001
            print(f"[meta] unreadable chin_metadata.json ({e}); using auto-detect")
    return None


CHIN_LANG_TOKEN = _chin_lang_token(MODEL_NAME)
print(f"  language token: {CHIN_LANG_TOKEN or 'auto-detect (V5-era model)'}")


# ---------------- Chin → English ----------------
def to_en(text_chin: str) -> str:
    """Translate Hakha Chin → English via Google's endpoint with the source pinned.
    deep-translator's language list lacks cnh and autodetect misreads it, so we
    call the endpoint directly (same approach as the batch app)."""
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


# ---------------- English text → speech ----------------
def speak_en(text_en: str):
    """Synthesize English → (sample_rate, int16 PCM shape (1, N)) for FastRTC,
    or None. gTTS for now; swap to Piper for low-latency local synthesis."""
    text = (text_en or "").strip()
    if not text:
        return None
    try:
        from gtts import gTTS
        mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        try:
            gTTS(text=text, lang="en").save(mp3)
            audio, sr = librosa.load(mp3, sr=24000, mono=True)  # float32 in [-1, 1]
        finally:
            os.unlink(mp3)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).reshape(1, -1)
        return sr, pcm
    except Exception as e:
        print(f"[tts] failed ({e})")
        return None


# ---------------- The real-time turn handler ----------------
def on_utterance(audio):
    """Called by ReplyOnPause when the speaker pauses.

    audio = (sample_rate, np.ndarray) of the captured Chin utterance. We
    transcribe → translate → speak, and yield the English audio back to the ear.
    """
    sr, samples = audio
    samples = np.asarray(samples).astype(np.float32).flatten()
    # FastRTC usually hands int16-range values; normalize to [-1, 1] for Whisper.
    peak = np.max(np.abs(samples)) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / 32768.0
    if sr != 16000:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)

    lang_kw = {"language": CHIN_LANG_TOKEN} if CHIN_LANG_TOKEN else {}
    segs, _ = MODEL.transcribe(samples, task="transcribe", beam_size=5,
                               vad_filter=False, **lang_kw)
    chin = "".join(s.text for s in segs).strip()
    if not chin:
        return
    english = to_en(chin)
    print(f"CHIN: {chin!r}  →  EN: {english!r}", flush=True)

    out = speak_en(english)
    if out is not None:
        yield out  # (sample_rate, int16 PCM) → played into the earbud


# ---------------- Wire up the stream ----------------
# WebRTC across networks (phone on cellular → Colab) needs a TURN relay.
# Preferred: STATIC TURN creds via env (TURN_URLS / TURN_USERNAME /
# TURN_CREDENTIAL), same scheme as the Space app — the HF/Cloudflare credential
# broker is the part that keeps DNS-failing. Fallback: broker via HF_TOKEN.
# With neither, None still works on localhost / same LAN. See REALTIME.md.
rtc_configuration = None
_hf = os.environ.get("HF_TOKEN")
_turn_urls = os.environ.get("TURN_URLS")
if _turn_urls:
    rtc_configuration = {"iceServers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {
            "urls": [u.strip() for u in _turn_urls.split(",") if u.strip()],
            "username": os.environ.get("TURN_USERNAME", ""),
            "credential": os.environ.get("TURN_CREDENTIAL", ""),
        },
    ]}
    print("[turn] using STATIC TURN from env (TURN_URLS)")
elif _hf:
    try:
        # get_hf_turn_credentials is deprecated (and its old endpoint 404s/DNS-fails);
        # FastRTC now brokers free Cloudflare TURN from your HF token.
        from fastrtc import get_cloudflare_turn_credentials  # type: ignore
        try:
            rtc_configuration = get_cloudflare_turn_credentials(hf_token=_hf)
        except TypeError:
            rtc_configuration = get_cloudflare_turn_credentials(token=_hf)
        print("[turn] using Cloudflare TURN credentials")
    except Exception as e:
        print(f"[turn] TURN setup failed ({e}); cross-network will likely fail.")
else:
    print("[turn] neither TURN_URLS nor HF_TOKEN set; cross-network will likely fail.")

stream = Stream(
    handler=ReplyOnPause(on_utterance),
    modality="audio",
    mode="send-receive",
    rtc_configuration=rtc_configuration,
)

if __name__ == "__main__":
    print("\n🚀 Launching real-time interpreter…")
    # .ui gives a Gradio-style page with a phone-openable share link.
    stream.ui.launch(server_name="0.0.0.0", share=True)
