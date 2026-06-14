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
from collections import deque

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

CHIN_CODE = "cnh"

# Skip translating/speaking when the input is detected as English. Besides being
# the requested behavior, this breaks the TTS→mic feedback loop (the English we
# speak won't get re-transcribed and echoed). Tunable via env: SKIP_ENGLISH=0
# disables it; EN_SKIP_PROB sets the language-confidence gate.
SKIP_ENGLISH = os.environ.get("SKIP_ENGLISH", "1") != "0"
EN_SKIP_PROB = float(os.environ.get("EN_SKIP_PROB", "0.5"))


def translate(text: str, sl: str, tl: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    import urllib.parse
    import urllib.request
    url = ("https://translate.googleapis.com/translate_a/single"
           f"?client=gtx&sl={sl}&tl={tl}&dt=t&q=" + urllib.parse.quote(text))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
        return "".join(seg[0] for seg in data[0] if seg and seg[0]).strip() or text
    except Exception as e:
        print(f"[translate] failed ({e})")
        return text


def speak(text: str, lang: str):
    """TTS via gTTS. Returns None if the language is unsupported (e.g. Chin),
    in which case the caller just shows text without audio."""
    text = (text or "").strip()
    if not text:
        return None
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
        print(f"[tts] failed for lang={lang!r} ({e}) — showing text only")
        return None


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
    "cnh2en": {"asr_lang": None, "sl": CHIN_CODE, "tl": "en", "tts": "en"},
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
_dn = {"model": None, "state": None, "sr": 48000, "ready": False, "warned": False}


def denoise(samples, sr):
    """samples: float32 mono in [-1, 1] at `sr`. Returns (clean_samples, sr_out).

    DeepFilterNet runs at 48 kHz so it may change the rate; the caller resamples
    to 16 kHz afterward regardless.
    """
    if DENOISE in ("off", "0", "none", "") or samples.size == 0:
        return samples, sr
    try:
        if DENOISE == "df":
            if not _dn["ready"]:
                from df.enhance import init_df
                _dn["model"], _dn["state"], _ = init_df()
                _dn["sr"] = _dn["state"].sr()
                _dn["ready"] = True
            import torch as _torch
            from df.enhance import enhance
            tgt = _dn["sr"]
            audio = (samples if sr == tgt
                     else librosa.resample(samples, orig_sr=sr, target_sr=tgt))
            clean = enhance(_dn["model"], _dn["state"],
                            _torch.from_numpy(audio).unsqueeze(0))
            return clean.squeeze(0).cpu().numpy().astype(np.float32), tgt
        # default: noisereduce spectral gating (non-stationary adapts to changing noise)
        import noisereduce as nr
        clean = nr.reduce_noise(y=samples, sr=sr, prop_decrease=DENOISE_AMOUNT,
                                stationary=False)
        return clean.astype(np.float32), sr
    except Exception as e:  # noqa: BLE001
        if not _dn["warned"]:
            print(f"[denoise] backend {DENOISE!r} unavailable ({e}); "
                  f"passing audio through.", flush=True)
            _dn["warned"] = True
        return samples, sr


def on_utterance(audio):
    sr, samples = audio
    samples = np.asarray(samples).astype(np.float32).flatten()
    peak = np.max(np.abs(samples)) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / 32768.0
    # Suppress ambient noise on the native-rate signal, before gain/downsample
    # (boosting cleaned speech beats boosting the noise along with it).
    samples, sr = denoise(samples, sr)
    gain = SETTINGS["gain"]
    if gain != 1.0:
        samples = np.clip(samples * gain, -1.0, 1.0)
    if sr != 16000:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
    cfg = DIRECTIONS[SETTINGS["direction"]]
    asr_kwargs = {"task": "transcribe", "beam_size": 5, "vad_filter": False}
    if cfg["asr_lang"]:
        asr_kwargs["language"] = cfg["asr_lang"]
    segs, info = MODEL.transcribe(samples, **asr_kwargs)
    src_text = "".join(s.text for s in segs).strip()
    if not src_text:
        return
    lang = getattr(info, "language", "") or ""
    lang_prob = getattr(info, "language_probability", 0.0) or 0.0
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
    ui_args={"title": "Hahka Chin Audio Translator"},
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
        " padding: var(--size-2) var(--size-4) !important; }"
        "</style>"
    )
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
