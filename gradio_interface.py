# ===============================
# Hakha Chin STT → EN (VAD + FW)
# Keeps your existing Gradio UI
# ===============================

# Optional: install (safe to re-run on Colab)
import sys, subprocess, os, json, tempfile, string
def pipi(*args): subprocess.run([sys.executable, "-m", "pip", "install", *args], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
pipi("faster-whisper", "torchaudio", "gradio", "deep-translator")

import torch, torchaudio, numpy as np, gradio as gr
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

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
                    model_name="large-v3",
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

    model = WhisperModel(model_name,
                         device=DEVICE,
                         compute_type="float16" if DEVICE=="cuda" else "int8")

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

# EN→CNH for the first box so UI remains “Hakha Chin → English”
def to_cnh(text_en: str) -> str:
    if not text_en: return ""
    try:
        return GoogleTranslator(source="en", target="cnh").translate(text_en)
    except Exception:
        # worst case: show English if cnh target is unavailable
        return text_en

def process_audio(audio_file: str):
    if not audio_file:
        return "❌ Upload audio!", "", ""
    try:
        refined, english = transcribe_file(audio_file, model_name="large-v3")
        # If large-v3 OOMs on T4, switch to "medium"
        # refined, english = transcribe_file(audio_file, model_name="medium")

        chin_display = to_cnh(english)               # keep first box as “Hakha Chin”
        stats = f"**Device:** {DEVICE.upper()} | **Segments:** {len(refined)} | **Model:** large-v3"
        return chin_display or "(empty)", english or "(empty)", stats
    except Exception as e:
        return f"❌ {e}", "", ""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎤 Hakha Chin Speech-to-Text (Optimized Backend)\nUpload audio → Hakha Chin (display) → English")
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio Input")
            btn = gr.Button("Translate", variant="primary")
        with gr.Column():
            transcription_out = gr.Textbox(label="📝 Hakha Chin", lines=10)
            translation_out = gr.Textbox(label="🌍 English", lines=10)
            stats_out = gr.Markdown()
    btn.click(fn=process_audio, inputs=audio_input, outputs=[transcription_out, translation_out, stats_out])
    audio_input.change(fn=process_audio, inputs=audio_input, outputs=[transcription_out, translation_out, stats_out])

print("\n🚀 Launching…")
demo.launch(share=True, debug=True)