# ===============================
# Hakha Chin STT → EN (VAD + FW)
# Keeps your existing Gradio UI
# ===============================

# Optional: install (safe to re-run on Colab)
import sys, subprocess, os, json, tempfile, string
def pipi(*args): subprocess.run([sys.executable, "-m", "pip", "install", *args], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
pipi("faster-whisper", "torchaudio", "gradio", "deep-translator", "gTTS")

import torch, torchaudio, numpy as np, gradio as gr
from pathlib import Path
from faster_whisper import WhisperModel, available_models
from deep_translator import GoogleTranslator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Default is the stock large-v3 (works before fine-tuning). After training +
# export_model.py, set CHIN_MODEL to the local CT2 dir, e.g.:
#   CHIN_MODEL=whisper-cnh-turbo-ct2 python gradio_interface.py
MODEL_NAME = os.environ.get("CHIN_MODEL", "large-v3")


def resolve_model(name):
    """Turn CHIN_MODEL into something WhisperModel can actually load.

    faster-whisper only accepts: a built-in size name (large-v3, turbo, ...),
    an HF repo id ("org/model"), or an *existing* local directory. If you pass a
    bare local name like 'whisper-cnh-turbo-ct2' and the folder isn't found
    (e.g. the app is launched from a different working dir than export_model.py
    wrote to), it silently treats it as a size name and dies with a confusing
    'Invalid model size' error. Resolve to an absolute path here, and fail fast
    with an actionable message instead.
    """
    # Existing local dir → pin to absolute path so cwd can't break it later.
    p = Path(name).expanduser()
    if p.is_dir():
        return str(p.resolve())
    # Built-in size → let faster-whisper download it.
    if name in available_models():
        return name
    # HF repo id → let faster-whisper resolve/download it.
    if "/" in name:
        return name
    # Looks like a local CT2 dir name but the folder isn't here.
    raise FileNotFoundError(
        f"CHIN_MODEL='{name}' is not a built-in Whisper size and no such "
        f"directory exists (looked in {p.resolve().parent}). "
        f"Run export_model.py to produce the CT2 folder, then launch from the "
        f"directory that contains it (or set CHIN_MODEL to its absolute path). "
        f"Built-in sizes: {', '.join(available_models())}."
    )


MODEL_PATH = resolve_model(MODEL_NAME)
# Loud banner so it's impossible to mistake which model is actually serving
# (e.g. the stale V4 in an old folder vs. the fine-tuned CT2 model).
print("=" * 64)
print(f"  Hakha Chin STT  |  MODEL: {MODEL_NAME}  |  DEVICE: {DEVICE}")
if MODEL_PATH != MODEL_NAME:
    print(f"  resolved → {MODEL_PATH}")
if MODEL_NAME == "large-v3":
    print("  ⚠️  stock large-v3 (NOT fine-tuned). Set CHIN_MODEL=whisper-cnh-turbo-ct2")
print("=" * 64)

# Load once at startup: surfaces a bad model immediately (not on first request)
# and avoids re-reading the weights on every audio upload.
print(f"⏳ Loading model ({MODEL_PATH}) ...")
MODEL = WhisperModel(MODEL_PATH,
                     device=DEVICE,
                     compute_type="float16" if DEVICE == "cuda" else "int8")
print("✓ Model loaded.")

# ---------------- Core helpers ----------------

def silero_vad_windows(audio_path, sr_target=16000, min_sil_ms=500, pad_ms=140, merge_gap_ms=350):
    wav, sr = torchaudio.load(audio_path)
    wav = wav.mean(dim=0, keepdim=True)  # mono
    if sr != sr_target:
        wav = torchaudio.functional.resample(wav, sr, sr_target)
    sr = sr_target
    samples = wav.squeeze(0)

    torch.hub.set_dir('/content/torchhub')  # harmless outside Colab
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    get_speech_timestamps = utils[0]

    ts = get_speech_timestamps(samples, model, sampling_rate=sr,
                               min_silence_duration_ms=min_sil_ms,
                               speech_pad_ms=pad_ms)

    # merge close chunks (compressed audio creates micro gaps)
    merged = []
    for t in ts:
        if not merged or (t["start"] - merged[-1]["end"]) > int(merge_gap_ms * sr / 1000):
            merged.append(t.copy())
        else:
            merged[-1]["end"] = t["end"]
    return [(t["start"]/sr, t["end"]/sr) for t in merged]

def slice_to_wav(full_path, start_s, end_s):
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run([
        "ffmpeg", "-y", "-i", full_path, "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
        "-ac", "1", "-ar", "16000", out
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out

def looks_garbage(txt, non_ascii_ratio=0.30, min_ascii_chars=12):
    t = (txt or "").strip()
    if not t: return True
    n_non_ascii = sum(1 for ch in t if ch not in string.printable)
    n_ascii     = len(t) - n_non_ascii
    return (n_non_ascii / len(t) > non_ascii_ratio) or (n_ascii < min_ascii_chars)

def refine_by_word_gaps(segments, max_gap_s=0.60, max_len_s=20.0, min_len_s=1.0):
    out, buf = [], None
    def flush():
        nonlocal buf
        if not buf: return
        text = "".join(w["word"] for w in buf).strip()
        if text:
            out.append({"start": buf[0]["start"], "end": buf[-1]["end"], "text": text})
        buf = None

    for seg in segments:
        for w in (seg.get("words") or []):
            if buf is None:
                buf = [w]; continue
            gap = w["start"] - buf[-1]["end"]
            dur = w["end"]   - buf[0]["start"]
            if gap > max_gap_s or dur > max_len_s:
                flush(); buf = [w]
            else:
                buf.append(w)
    flush()

    # merge tiny stubs forward
    res = []
    for s in out:
        if res and (s["end"] - s["start"]) < min_len_s:
            res[-1]["end"]  = s["end"]
            res[-1]["text"] = (res[-1]["text"] + " " + s["text"]).strip()
        else:
            res.append(s)
    return res

# ---------------- Transcribe pipeline ----------------

def transcribe_file(audio_path,
                    model=MODEL,
                    max_subwin_s=20.0,
                    vad_min_sil_ms=500,
                    vad_pad_ms=140,
                    vad_merge_gap_ms=350,
                    fallback_to_en=True):
    """
    Returns: (refined_segments, english_transcript)
    """
    wins = silero_vad_windows(audio_path,
                              min_sil_ms=vad_min_sil_ms,
                              pad_ms=vad_pad_ms,
                              merge_gap_ms=vad_merge_gap_ms)

    segments = []
    for (s0, s1) in wins:
        t = s0
        while t < s1:
            u = min(t + max_subwin_s, s1)
            tmp = slice_to_wav(audio_path, t, u)
            try:
                # First try same-language transcribe
                part, _ = model.transcribe(
                    tmp,
                    task="transcribe",
                    condition_on_previous_text=False,
                    word_timestamps=True,
                    beam_size=5, best_of=5, patience=1.0,
                    temperature=[0.0, 0.2, 0.4],
                    compression_ratio_threshold=2.2,
                    vad_filter=False
                )
                # faster-whisper returns a ONE-SHOT generator. Materialize it, or
                # the text scan below exhausts it and the segment loop sees nothing.
                part = list(part)
                txt = "".join(s.text for s in part).strip()

                # If glyph soup → fallback to English translate
                if fallback_to_en and looks_garbage(txt):
                    part, _ = model.transcribe(
                        tmp,
                        task="translate",
                        language="en",
                        condition_on_previous_text=False,
                        word_timestamps=True,
                        beam_size=5, best_of=5, patience=1.0,
                        temperature=[0.0, 0.2, 0.4],
                        compression_ratio_threshold=2.2,
                        vad_filter=False
                    )
                    part = list(part)

                for s in part:
                    segments.append({
                        "start": s.start + t,
                        "end":   s.end   + t,
                        "text":  s.text,
                        "words": [{"start": w.start + t, "end": w.end + t, "word": w.word}
                                  for w in (s.words or [])]
                    })
            finally:
                os.unlink(tmp)
            t = u

    segments.sort(key=lambda x: x["start"])
    refined = refine_by_word_gaps(segments, max_gap_s=0.60, max_len_s=20.0)
    english = " ".join(s["text"] for s in refined).strip()
    return refined, english

# --------------- Gradio wiring (UI unchanged) ---------------

# Hakha Chin → English. The fine-tuned model outputs Chin text (task="transcribe"),
# so we translate Chin→EN here. deep-translator's bundled language list is stale and
# does NOT include Hakha Chin, and Google's autodetect misreads it (→ "Krio"/garbage).
# But Google's endpoint itself honors sl=cnh, so we call it directly with the source
# pinned. (Verified: "Na na maw?" → "Are you?")
CHIN_CODE = "cnh"

def _google_translate(text: str, sl: str, tl: str = "en") -> str:
    import urllib.parse, urllib.request
    url = ("https://translate.googleapis.com/translate_a/single"
           f"?client=gtx&sl={sl}&tl={tl}&dt=t&q=" + urllib.parse.quote(text))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8"))
    # data[0] is a list of [translated_chunk, original_chunk, ...]
    return "".join(seg[0] for seg in data[0] if seg and seg[0]).strip()

def to_en(text_chin: str) -> str:
    if not text_chin: return ""
    try:
        out = _google_translate(text_chin, CHIN_CODE, "en")
        return out or text_chin
    except Exception as e:
        print(f"[translate] {CHIN_CODE}->en failed ({e}); showing Chin instead.")
        return text_chin

def speak_en(text_en: str):
    """Synthesize English speech (mp3) from the translation. Returns a filepath
    for the gr.Audio output, or None if there's nothing to say."""
    text = (text_en or "").strip()
    if not text or text == "(empty)":
        return None
    try:
        from gtts import gTTS
        out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        gTTS(text=text, lang="en").save(out)
        return out
    except Exception as e:
        print(f"[tts] failed ({e}); no audio.")
        return None

def process_audio(audio_file: str):
    if not audio_file:
        return "❌ Upload audio!", "", "", None
    try:
        # transcribe_file returns (segments, transcription). The fine-tuned model
        # transcribes Hakha Chin, so this text is Chin — translate it to English.
        refined, chin = transcribe_file(audio_file, fallback_to_en=False)
        english = to_en(chin)
        stats = f"**Device:** {DEVICE.upper()} | **Segments:** {len(refined)} | **Model:** {MODEL_NAME}"
        return chin or "(empty)", english or "(empty)", stats, speak_en(english)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("\n===== process_audio ERROR =====\n" + tb, flush=True)
        detail = f"❌ {type(e).__name__}: {e}\n\n{tb}"
        return detail, detail, "**Error** — full traceback in the boxes and the cell logs", None

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎤 Hakha Chin Speech-to-Text (Optimized Backend)\nUpload audio → Hakha Chin (display) → English")
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio Input")
            btn = gr.Button("Translate", variant="primary")
        with gr.Column():
            transcription_out = gr.Textbox(label="📝 Hakha Chin", lines=10)
            translation_out = gr.Textbox(label="🌍 English", lines=10)
            audio_out = gr.Audio(label="🔊 English (spoken)", autoplay=True)
            stats_out = gr.Markdown()
    _outs = [transcription_out, translation_out, stats_out, audio_out]
    # Pressing Translate while recording should also STOP the recording. The mic is
    # a frontend widget, so we click its Stop button via JS; stop_recording (below)
    # then fires with the finalized audio and runs the pipeline.
    _stop_js = """
    (audio) => {
        const b = document.querySelector('button[aria-label="Stop recording"]')
              || [...document.querySelectorAll('button')].find(
                   el => /stop/i.test((el.getAttribute('aria-label')||'') + ' ' + (el.textContent||'')));
        if (b) b.click();
        return audio;
    }
    """
    btn.click(fn=process_audio, inputs=audio_input, outputs=_outs, js=_stop_js)
    # Auto-run when a recording stops or a file is uploaded (one fire each, no dupes).
    audio_input.stop_recording(fn=process_audio, inputs=audio_input, outputs=_outs)
    audio_input.upload(fn=process_audio, inputs=audio_input, outputs=_outs)

print("\n🚀 Launching…")
demo.launch(share=True, debug=True)