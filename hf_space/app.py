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
import re
import tempfile
import functools
from collections import OrderedDict, deque

import numpy as np
import librosa
import torch
import gradio as gr
from faster_whisper import WhisperModel
from fastrtc import (
    Stream, ReplyOnPause, AlgoOptions, SileroVadOptions, AdditionalOutputs,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CHIN_MODEL must be an HF *model-repo id* for the fine-tuned CT2 model (the Space
# runs on HF infra and can't read a Drive path). faster-whisper downloads it.
#
# Hard-fail when it's unset instead of silently loading stock Whisper. Stock
# whisper-large-v3 has NO Hakha Chin training, so it produces garbage Chin
# transcripts that look like a translation bug — this exact misconfiguration
# (CHIN_MODEL unset → stock fallback) silently broke the Space before. Refusing
# to start surfaces the problem in the Space logs immediately. Set ALLOW_STOCK=1
# only to intentionally demo the base model. Use hf_space/upload_model.py to
# publish your CT2 model and get the repo id to set here.
MODEL_NAME = os.environ.get("CHIN_MODEL")
if not MODEL_NAME:
    if os.environ.get("ALLOW_STOCK") == "1":
        MODEL_NAME = "openai/whisper-large-v3"
        print("⚠️  CHIN_MODEL unset and ALLOW_STOCK=1 → serving STOCK "
              "openai/whisper-large-v3, which has NO Hakha Chin training. "
              "Transcripts WILL be garbage; this is for base-model demos only.")
    else:
        raise SystemExit(
            "❌ CHIN_MODEL is not set. This Space must point at your fine-tuned "
            "Hakha Chin CT2 model repo — set the Space Variable\n"
            "    CHIN_MODEL = <user>/whisper-cnh-turbo-ct2\n"
            "(publish the model with hf_space/upload_model.py first). Refusing to "
            "start on stock Whisper, which knows no Hakha Chin and silently emits "
            "garbage. Set ALLOW_STOCK=1 only to intentionally demo the base model."
        )
print(f"Loading model: {MODEL_NAME} on {DEVICE}")
MODEL = WhisperModel(MODEL_NAME, device=DEVICE,
                     compute_type="float16" if DEVICE == "cuda" else "int8")
print("✓ Model loaded.")

# English ASR for the English→Chin direction. The cnh-fine-tuned model above
# mangles English ("street" → "strih") because it specialized on Hakha Chin, so
# transcribe English with a stock English Whisper instead, then translate en→cnh.
# Loaded lazily on first en→cnh use, so it costs nothing if that direction is
# never used. EN_ASR_MODEL = any faster-whisper size (default small.en: light +
# accurate English).
EN_ASR_MODEL = os.environ.get("EN_ASR_MODEL", "small.en")
_en_asr = {"model": None, "tried": False}


def english_asr_model():
    """Stock English Whisper for the en→cnh ASR step (cached). Falls back to the
    fine-tuned model if it can't load."""
    if _en_asr["tried"]:
        return _en_asr["model"]
    _en_asr["tried"] = True
    try:
        _en_asr["model"] = WhisperModel(
            EN_ASR_MODEL, device=DEVICE,
            compute_type="float16" if DEVICE == "cuda" else "int8")
        print(f"✓ English ASR model loaded: {EN_ASR_MODEL}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[asr] English model '{EN_ASR_MODEL}' unavailable ({e}); "
              f"using the fine-tuned model for English.", flush=True)
        _en_asr["model"] = None
    return _en_asr["model"]


CHIN_CODE = "cnh"


def _chin_lang_token(model_ref: str):
    """Surrogate language token the V6+ adapters were trained with (see
    train.py in the main repo). Whisper has no cnh token, so training fixes a
    stand-in (e.g. "id") and inference must force the SAME one — auto-detect
    feeds a different, never-trained prompt per utterance. export_model.py
    puts chin_metadata.json in the CT2 dir and upload_model.py ships it with
    the repo. CHIN_LANG env overrides (CHIN_LANG="" → auto-detect). None for
    V5-era models → old auto-detect behavior."""
    if "CHIN_LANG" in os.environ:
        return os.environ["CHIN_LANG"] or None
    try:
        from pathlib import Path
        if Path(model_ref).is_dir():
            meta_path = Path(model_ref) / "chin_metadata.json"
            if not meta_path.is_file():
                return None
        else:  # HF repo id — the file rides in the model repo
            from huggingface_hub import hf_hub_download
            meta_path = hf_hub_download(model_ref, "chin_metadata.json",
                                        token=os.environ.get("HF_TOKEN") or None)
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f).get("language_token")
    except Exception as e:  # noqa: BLE001
        print(f"[meta] no chin_metadata.json for {model_ref!r} ({e}); "
              f"using language auto-detect (V5-era model).", flush=True)
        return None


CHIN_LANG_TOKEN = _chin_lang_token(MODEL_NAME)
print(f"  cnh ASR language token: {CHIN_LANG_TOKEN or 'auto-detect'}", flush=True)

# Version is defined HERE, in source — the deployed code IS the version. No env
# override on purpose: a Space variable can't silently pin the build stamp, so
# the "Build:" you see always equals the code that's running. Bump this when you
# ship a change.
APP_VERSION = "v5.3.0-piper"
print(f"[app] version={APP_VERSION}", flush=True)

# Skip translating/speaking when the input is detected as English. Besides being
# the requested behavior, this breaks the TTS→mic feedback loop (the English we
# speak won't get re-transcribed and echoed). Tunable via env: SKIP_ENGLISH=0
# disables it; EN_SKIP_PROB sets the language-confidence gate.
SKIP_ENGLISH = os.environ.get("SKIP_ENGLISH", "1") != "0"
EN_SKIP_PROB = float(os.environ.get("EN_SKIP_PROB", "0.5"))


# --- Caching -----------------------------------------------------------------
# People repeat phrases ("yes", "okay", "how are you"), and each repeat otherwise
# re-hits Google Translate AND regenerates TTS audio. Cache both on a normalized
# text key to cut latency, network calls, and battery. Single-user Space, so
# in-process LRUs are fine and nothing is persisted (no disk needed).
# CACHE_LOG=1 logs hits/misses; TTS_CACHE_MAX bounds the audio cache.
CACHE_LOG = os.environ.get("CACHE_LOG", "0") == "1"


def _norm(text: str) -> str:
    """Trim + collapse internal whitespace, for stable cache keys."""
    return re.sub(r"\s+", " ", (text or "").strip())


@functools.lru_cache(maxsize=512)
def _translate_cached(norm_text: str, sl: str, tl: str) -> str:
    """The Google call, on normalized text. Raises on failure so failures are
    NOT cached — the wrapper catches and falls back without poisoning the cache."""
    import urllib.parse
    import urllib.request
    url = ("https://translate.googleapis.com/translate_a/single"
           f"?client=gtx&sl={sl}&tl={tl}&dt=t&q=" + urllib.parse.quote(norm_text))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode("utf-8"))
    return "".join(seg[0] for seg in data[0] if seg and seg[0]).strip()


def translate(text: str, sl: str, tl: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    norm = _norm(text)
    before = _translate_cached.cache_info().hits
    try:
        out = _translate_cached(norm, sl, tl)
    except Exception as e:  # noqa: BLE001
        print(f"[translate] failed ({e})")
        return text
    if CACHE_LOG:
        hit = _translate_cached.cache_info().hits > before
        print(f"[cache] translate {'HIT' if hit else 'miss'}: {norm[:40]!r}", flush=True)
    return out or text


# --- Text-to-speech ----------------------------------------------------------
# TTS_BACKEND=piper (default) synthesizes English locally with Piper — no
# per-phrase network round-trip or MP3 temp file, much lower latency than gTTS,
# and CPU-light. gTTS is the automatic fallback on ANY Piper failure (dep not
# installed, model missing, synth error), so the app always boots and speaks.
# Set TTS_BACKEND=gtts to force the old path. Piper only covers English (cnh→en);
# Chin has no voice in either backend and stays text-only. speak() caches the
# result of whichever backend runs.
#
# Piper voice: PIPER_MODEL = a local .onnx path (with PIPER_CONFIG or a sibling
# .onnx.json), else PIPER_VOICE names a voice downloaded from rhasspy/piper-voices.
TTS_BACKEND = os.environ.get("TTS_BACKEND", "piper").lower()
PIPER_VOICE = os.environ.get("PIPER_VOICE", "en_US-lessac-medium")
_piper = {"voice": None, "tried": False, "warned": False}

# TTS cache: (normalized_text, lang) → (sample_rate, int16 PCM). Only successful
# audio is cached (never None, so failures / Chin-text-only keep retrying). Store
# and return COPIES so a consumer can never mutate the cached array.
_TTS_CACHE_MAX = int(os.environ.get("TTS_CACHE_MAX", "256"))
_tts_cache = OrderedDict()


def _load_piper():
    """Load the Piper voice once (cached). Returns the voice or None (→ gTTS)."""
    if _piper["tried"]:
        return _piper["voice"]
    _piper["tried"] = True
    try:
        from piper import PiperVoice
        pm = os.environ.get("PIPER_MODEL", "").strip()
        if pm and os.path.isfile(pm):
            onnx = pm
            conf = os.environ.get("PIPER_CONFIG", "").strip() or pm + ".json"
        else:
            # rhasspy/piper-voices layout: <lang>/<lang_region>/<name>/<quality>/<full>
            from huggingface_hub import hf_hub_download
            parts = PIPER_VOICE.split("-")            # e.g. ["en_US","lessac","medium"]
            base = f"{parts[0].split('_')[0]}/{parts[0]}/{parts[1]}/{parts[2]}/{PIPER_VOICE}"
            onnx = hf_hub_download("rhasspy/piper-voices", base + ".onnx")
            conf = hf_hub_download("rhasspy/piper-voices", base + ".onnx.json")
        _piper["voice"] = PiperVoice.load(onnx, config_path=conf)
        print(f"✓ Piper voice loaded: {PIPER_VOICE}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[tts] Piper unavailable ({e}); using gTTS.", flush=True)
        _piper["voice"] = None
    return _piper["voice"]


def _speak_piper(text: str):
    """English → (sample_rate, int16 PCM (1, N)) via Piper, or None on failure.
    Handles both the 1.2.x (synthesize_stream_raw → bytes) and 1.3+
    (synthesize → AudioChunk) APIs."""
    voice = _load_piper()
    if voice is None:
        return None
    try:
        raw, sr = b"", None
        try:  # newer API (piper-tts ≥1.3): iterable of AudioChunk
            chunks = list(voice.synthesize(text))
            if chunks and hasattr(chunks[0], "audio_int16_bytes"):
                raw = b"".join(c.audio_int16_bytes for c in chunks)
                sr = chunks[0].sample_rate
        except Exception:
            raw = b""
        if not raw:  # older API (piper-tts 1.2.x)
            raw = b"".join(voice.synthesize_stream_raw(text))
            sr = voice.config.sample_rate
        if not raw:
            return None
        return sr, np.frombuffer(raw, dtype=np.int16).reshape(1, -1)
    except Exception as e:  # noqa: BLE001
        if not _piper["warned"]:
            print(f"[tts] Piper synth failed ({e}); falling back to gTTS.", flush=True)
            _piper["warned"] = True
        return None


def _speak_gtts(text: str, lang: str):
    """TTS via gTTS. Returns (sr, int16 PCM) or None (unsupported lang, e.g. Chin)."""
    try:
        from gtts import gTTS
        mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        try:
            gTTS(text=text, lang=lang).save(mp3)
            audio, sr = librosa.load(mp3, sr=24000, mono=True)
        finally:
            os.unlink(mp3)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).reshape(1, -1)
        return sr, pcm
    except Exception as e:
        print(f"[tts] gTTS failed for lang={lang!r} ({e}) — showing text only")
        return None


def _synthesize(text: str, lang: str):
    """Dispatch to a backend: Piper for English when TTS_BACKEND=piper (with gTTS
    fallback on failure), else gTTS. Returns (sr, int16 PCM) or None."""
    if TTS_BACKEND == "piper" and lang == "en":
        out = _speak_piper(text)
        if out is not None:
            return out  # else fall through to gTTS
    return _speak_gtts(text, lang)


def speak(text: str, lang: str):
    """Cached TTS. Returns (sr, int16 PCM) or None (text-only). Synthesizes via
    _synthesize (Piper/gTTS), caches only successful audio, and hands back a copy
    so the cached array is never mutated."""
    text = (text or "").strip()
    if not text:
        return None
    key = (_norm(text), lang)
    hit = _tts_cache.get(key)
    if hit is not None:
        _tts_cache.move_to_end(key)
        if CACHE_LOG:
            print(f"[cache] tts HIT: {key[0][:40]!r}", flush=True)
        sr, pcm = hit
        return sr, pcm.copy()
    out = _synthesize(text, lang)
    if out is not None:
        sr, pcm = out
        _tts_cache[key] = (sr, pcm.copy())          # private copy in the cache
        _tts_cache.move_to_end(key)
        while len(_tts_cache) > _TTS_CACHE_MAX:
            _tts_cache.popitem(last=False)
        if CACHE_LOG:
            print(f"[cache] tts miss→store: {key[0][:40]!r}", flush=True)
    return out


# --- Mic sensitivity ---------------------------------------------------------
# AirPods (and other Bluetooth headsets) feed a quiet, AGC'd signal that the
# Silero VAD often misses, so utterances never trigger. A single "sensitivity"
# slider drives two knobs together: (1) it lowers the VAD speech threshold so
# quieter speech is detected, and (2) it applies a matching input gain so the
# boosted audio still transcribes cleanly. ReplyOnPause.copy() shares these
# option objects with every per-connection copy, so mutating them live takes
# effect on the active stream.
SETTINGS = {"gain": 1.0, "direction": "cnh2en"}

# Per-direction config: ASR language hint, translate source/target codes, and
# TTS language. English→Chin is text-only in practice — gTTS has no Chin voice,
# so speak() returns None and we just show the transcript.
DIRECTIONS = {
    "cnh2en": {"asr_lang": CHIN_LANG_TOKEN, "sl": CHIN_CODE, "tl": "en", "tts": "en"},
    "en2cnh": {"asr_lang": "en", "sl": "en", "tl": CHIN_CODE, "tts": CHIN_CODE},
}

vad_options = SileroVadOptions(threshold=0.5)
algo_options = AlgoOptions()


def _apply_sensitivity(sensitivity: float):
    """Map a 0–100 sensitivity (50 = neutral) onto VAD threshold + input gain."""
    vad_options.threshold = float(
        np.clip(0.5 - (sensitivity - 50) / 100.0 * 0.4, 0.15, 0.85)
    )
    SETTINGS["gain"] = max(0.2, float(sensitivity) / 50.0)
    print(f"[mic] sensitivity={sensitivity:.0f} → vad_threshold="
          f"{vad_options.threshold:.2f} gain={SETTINGS['gain']:.2f}", flush=True)


# Lean slightly sensitive by default, since AirPods are the motivating case.
_DEFAULT_SENSITIVITY = 60
_apply_sensitivity(_DEFAULT_SENSITIVITY)


# --- Ambient-noise suppression ----------------------------------------------
# Clean steady background (fans, traffic, hum) before transcription. Real rooms
# are out-of-distribution vs the clean Common Voice the model trained on, so
# quieter input usually helps WER. Backend via env DENOISE:
#   noisereduce (default) — spectral gating, light, pure-Python, no model load
#   df                     — DeepFilterNet (neural, cleaner, fewer artifacts);
#                            requires adding `deepfilternet` to requirements.txt
#   off                    — passthrough
# DENOISE_AMOUNT (0–1, default 0.85) eases noisereduce's reduction to limit the
# musical-noise artifacts that could themselves confuse a clean-trained model.
# Any failure logs once and passes audio through — denoise never breaks a turn.
DENOISE = os.environ.get("DENOISE", "noisereduce").lower()
DENOISE_AMOUNT = float(os.environ.get("DENOISE_AMOUNT", "0.85"))
# Stationary mode estimates one noise profile for the whole clip — far faster
# than non-stationary and a good fit for steady ambient noise (fans/hum/traffic).
# Set DENOISE_STATIONARY=0 for fluctuating noise (slower).
DENOISE_STATIONARY = os.environ.get("DENOISE_STATIONARY", "1") != "0"
_dn = {"model": None, "state": None, "sr": 48000, "ready": False, "warned": False}


def denoise(samples, sr):
    """Suppress ambient noise. samples: float32 mono in [-1, 1] at `sr`; returns
    cleaned samples at the SAME sr.

    Called on the 16 kHz signal the model sees: doing it at the 48 kHz native
    rate in non-stationary mode was slow enough to back the queue up indefinitely
    (the transcript rides additional_outputs on that queue, so it stalled). Any
    failure logs once and passes audio through — denoise never breaks a turn.
    """
    if DENOISE in ("off", "0", "none", "") or samples.size == 0:
        return samples
    try:
        if DENOISE == "df":
            if not _dn["ready"]:
                from df.enhance import init_df
                _dn["model"], _dn["state"], _ = init_df()
                _dn["sr"] = _dn["state"].sr()
                _dn["ready"] = True
            import torch as _torch
            from df.enhance import enhance
            tgt = _dn["sr"]  # DeepFilterNet runs at 48 kHz; up/downsample around it
            audio = (samples if sr == tgt
                     else librosa.resample(samples, orig_sr=sr, target_sr=tgt))
            clean = enhance(_dn["model"], _dn["state"],
                            _torch.from_numpy(audio).unsqueeze(0))
            clean = clean.squeeze(0).cpu().numpy().astype(np.float32)
            return (clean if sr == tgt
                    else librosa.resample(clean, orig_sr=tgt, target_sr=sr))
        import noisereduce as nr
        clean = nr.reduce_noise(y=samples, sr=sr, prop_decrease=DENOISE_AMOUNT,
                                stationary=DENOISE_STATIONARY)
        return clean.astype(np.float32)
    except Exception as e:  # noqa: BLE001
        if not _dn["warned"]:
            print(f"[denoise] backend {DENOISE!r} unavailable ({e}); "
                  f"passing audio through.", flush=True)
            _dn["warned"] = True
        return samples


def on_utterance(audio):
    sr, samples = audio
    samples = np.asarray(samples).astype(np.float32).flatten()
    peak = np.max(np.abs(samples)) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / 32768.0
    gain = SETTINGS["gain"]
    if gain != 1.0:
        samples = np.clip(samples * gain, -1.0, 1.0)
    if sr != 16000:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
        sr = 16000
    # Suppress ambient noise on the 16 kHz signal the model sees — cheap here;
    # at the 48 kHz native rate it was slow enough to back up the queue.
    samples = denoise(samples, sr)
    cfg = DIRECTIONS[SETTINGS["direction"]]
    # English→Chin: transcribe English with the stock English model (the
    # fine-tuned cnh model garbles English); translation to cnh happens below.
    # Chin→English keeps the fine-tuned model.
    if SETTINGS["direction"] == "en2cnh":
        asr = english_asr_model() or MODEL
    else:
        asr = MODEL
    asr_kwargs = {"task": "transcribe", "beam_size": 5, "vad_filter": False}
    if cfg["asr_lang"]:
        asr_kwargs["language"] = cfg["asr_lang"]
    segs, info = asr.transcribe(samples, **asr_kwargs)
    src_text = "".join(s.text for s in segs).strip()
    if not src_text:
        return
    lang = getattr(info, "language", "") or ""
    lang_prob = getattr(info, "language_probability", 0.0) or 0.0
    # When the cnh language token is FORCED (V6 models), transcribe() skips
    # detection and info.language just echoes the forced token — useless for
    # the English-echo check below. Run detection explicitly in that case.
    if (SETTINGS["direction"] == "cnh2en" and SKIP_ENGLISH
            and asr_kwargs.get("language")):
        try:
            lang, lang_prob, _ = asr.detect_language(audio=samples)
        except Exception as e:  # noqa: BLE001
            lang, lang_prob = "", 0.0
            print(f"[lang] detect_language unavailable ({e}); "
                  f"EN-skip check disabled this turn.", flush=True)
    print(f"[lang] dir={SETTINGS['direction']} detected={lang!r} p={lang_prob:.2f}", flush=True)
    # In Chin→English mode, when the input is English we skip only the AUDIO
    # output — this breaks the TTS→mic feedback loop (the English we'd speak
    # won't get re-transcribed and echoed). The text translation is shown
    # exactly as usual. (Not applied in English→Chin mode, where English is the
    # expected input.)
    skip_audio = (SETTINGS["direction"] == "cnh2en" and SKIP_ENGLISH
                  and lang == "en" and lang_prob >= EN_SKIP_PROB)
    out_text = translate(src_text, cfg["sl"], cfg["tl"])
    print(f"{cfg['sl'].upper()}: {src_text!r}  →  {cfg['tl'].upper()}: {out_text!r}", flush=True)
    # Push the text to the on-screen transcript first, so it appears even if TTS
    # fails or is skipped (e.g. no Chin voice, or English-input audio skip).
    yield AdditionalOutputs(src_text, out_text)
    if skip_audio:
        print(f"EN-SKIP (audio only): {src_text!r}", flush=True)
        return
    audio_out = speak(out_text, cfg["tts"])
    if audio_out is not None:
        yield audio_out


# TURN for Spaces. Per FastRTC's docs, pass the ASYNC Cloudflare credential
# function (HF brokers ~10 GB/mo free TURN via your HF token). It's invoked
# per-connection, so it satisfies the Spaces startup check WITHOUT a credential
# fetch at import time — the sync fetch was DNS-failing during startup.
_HF = os.environ.get("HF_TOKEN")
_TURN_URLS = os.environ.get("TURN_URLS")

server_rtc_configuration = None
if _TURN_URLS:
    # Preferred: STATIC TURN creds (set TURN_URLS / TURN_USERNAME / TURN_CREDENTIAL
    # as Space secrets, e.g. from a free ExpressTURN or Metered account). No
    # credential-broker fetch — which is the thing that kept DNS-failing.
    _static_turn = {"iceServers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {
            "urls": [u.strip() for u in _TURN_URLS.split(",") if u.strip()],
            "username": os.environ.get("TURN_USERNAME", ""),
            "credential": os.environ.get("TURN_CREDENTIAL", ""),
        },
    ]}
    rtc_configuration = _static_turn
    # The Space's server is ALSO behind NAT, so it needs a relay to be reachable
    # by the browser. Give both sides the same static TURN server.
    server_rtc_configuration = _static_turn
    print("[turn] using STATIC TURN from env (TURN_URLS) — client + server")
elif _HF:
    # Fallback: HF/Cloudflare async broker (DNS-fails in these sandboxes; kept
    # only so the app still boots without static creds).
    from fastrtc import get_cloudflare_turn_credentials_async

    async def rtc_configuration():
        return await get_cloudflare_turn_credentials_async(hf_token=_HF)

    print("[turn] no TURN_URLS set; falling back to Cloudflare broker (may fail)")
else:
    # Without creds the broker would just error per-connection — skip it. STUN
    # only: works on localhost / same LAN, not across NAT.
    rtc_configuration = None
    print("[turn] no TURN_URLS and no HF_TOKEN — WebRTC will only work on the same network")

# On-screen transcript. on_utterance yields AdditionalOutputs(chin, english);
# _update_transcript appends each turn to a server-side history (below) and
# returns the joined text, so the box scrolls instead of being replaced.
# Bounded so a long session doesn't grow the textbox value without limit.
_transcript_history = deque(maxlen=100)

transcript_box = gr.Textbox(
    label="📝 Transcript (source → translation)",
    value="",
    lines=12,
    max_lines=12,
    interactive=False,
    autoscroll=True,
    elem_id="transcript_box",
)


def _update_transcript(current: str, chin: str, english: str) -> str:
    # Accumulate the transcript server-side so it reliably scrolls (newest at
    # the bottom) instead of being replaced each turn. Relying on `current`
    # (the component value threaded back in) proved flaky over the WebRTC
    # stream, so we keep our own bounded history. Single-user Space, so a
    # module-level buffer is fine — same pattern as SETTINGS/vad_options.
    _transcript_history.append(f"🗣️ {chin}\n🔤 {english}")
    return "\n\n".join(_transcript_history)


stream = Stream(
    handler=ReplyOnPause(
        on_utterance,
        algo_options=algo_options,
        model_options=vad_options,
    ),
    modality="audio",
    mode="send-receive",
    additional_outputs=[transcript_box],
    additional_outputs_handler=_update_transcript,
    rtc_configuration=rtc_configuration,
    server_rtc_configuration=server_rtc_configuration,
    # Short, one-line title so it doesn't wrap and overlap the record button
    # on mobile.
    ui_args={"title": "Hakha Chin Audio Translator"},
)

# Spaces (gradio SDK) serves this `demo` object. Append a mic-sensitivity slider;
# its change handler mutates the shared VAD/gain settings used above, live.
demo = stream.ui

# FastRTC renders the transcript inside a right-hand gr.Sidebar that defaults to
# open, so the Space loads showing the (empty) transcript panel with the mic /
# Record controls tucked behind a chevron. Collapse that sidebar by default so
# the Record button is what you see on load; tap the chevron to slide the
# transcript back in. FastRTC builds the sidebar internally and exposes no flag
# for this, so we reach into the built Blocks and flip `open` before launch.
for _block in demo.blocks.values():
    if isinstance(_block, gr.Sidebar):
        _block.open = False

with demo:
    # Bump the transcript font size for readability. Also re-align the
    # full-screen WebRTC overlay (which holds the waveform + the Record button
    # in a full-height flex column) with space-between: waveform pinned just
    # under the title, Record button pinned to the bottom. We deliberately avoid
    # a CSS `transform` here — transforming the wave container offsets the
    # geometry its audio-reactive renderer relies on, which stops the
    # speech-driven animation.
    gr.HTML(
        "<style>"
        "#transcript_box textarea { font-size: 1.5rem !important;"
        " line-height: 1.6 !important; }"
        ".audio-container { justify-content: space-between !important;"
        " padding-bottom: 56px !important; }"
        ".audio-container .gradio-webrtc-waveContainer {"
        " margin-top: -34px !important; }"
        # Spread the in-call controls apart so a mobile thumb reaching for the
        # audio-mute doesn't land on Stop. .button-wrap is the flex pill holding
        # [Stop] [audio-mute] [mic-mute] (FastRTC 0.0.34); widen the gaps and
        # enlarge the mute tap-targets. gap/padding are tunable if it still feels
        # cramped (or too spread) on your phone.
        ".button-wrap.full-screen { gap: var(--size-7) !important;"
        " flex-wrap: wrap !important; justify-content: center !important; }"
        ".button-wrap .mute-button {"
        " padding: var(--size-2) var(--size-4) !important;"
        " border-radius: var(--radius-lg) !important; }"
        ".app-version { font-size: 0.85rem; opacity: 0.7; text-align: center;"
        " margin: -0.25rem 0 0.75rem; }"
        "</style>"
    )
    gr.HTML(f"<div class='app-version'>Build: {APP_VERSION}</div>")
    direction = gr.Radio(
        choices=[("Hakha Chin → English", "cnh2en"),
                 ("English → Hakha Chin", "en2cnh")],
        value="cnh2en",
        label="🔁 Translation direction",
    )
    direction.change(lambda d: SETTINGS.update(direction=d),
                     inputs=direction, outputs=None)
    sensitivity = gr.Slider(
        minimum=0, maximum=100, value=_DEFAULT_SENSITIVITY, step=5,
        label="🎙️ Mic sensitivity",
        info="Raise for AirPods / quiet mics: detects quieter speech and boosts input gain.",
    )
    sensitivity.change(_apply_sensitivity, inputs=sensitivity, outputs=None)

    # Default the audio OUTPUT to muted. The mute-audio button only exists once
    # the stream opens, so we can't click it at page load — instead watch the
    # DOM and click it the first time it appears unmuted (aria-label "mute
    # audio"). One-shot: after the auto-mute we disconnect, so if the user later
    # unmutes it stays unmuted. (FastRTC exposes no default-muted flag.)
    demo.load(
        fn=None,
        js="""
() => {
  let done = false;
  const mute = () => {
    if (done) return;
    const b = document.querySelector('button.mute-button[aria-label="mute audio"]');
    if (b) { b.click(); done = true; obs.disconnect(); }
  };
  const obs = new MutationObserver(mute);
  obs.observe(document.body, {subtree: true, childList: true,
    attributes: true, attributeFilter: ['aria-label']});
  mute();
}
""",
    )

if __name__ == "__main__":
    demo.launch()
