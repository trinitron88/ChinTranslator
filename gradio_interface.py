"""
Gradio Interface - FIXED VERSION (Option A, VAD+Gain+Beam)
- 16 kHz mono normalization (mic & uploads)
- Non-silence segmentation + lower gate for phone recordings
- 30s sliding-window transcription with overlap
- Dual-pass decoding (forced transcribe, then unforced fallback)
- Beam search & temperature 0 for stability
- Always attempts translation to English (with resilient fallback)
"""

import json, sys
import re
import numpy as np
import gradio as gr
import torch, librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from deep_translator import GoogleTranslator

print("🤖 Loading Hakha Chin Whisper Model...")
MODEL_PATH = "./whisper-hakha-chin"

processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Model loaded on {device}")

# soundfile optional
try:
    import soundfile as sf
except Exception:
    sf = None
    print("ℹ️ 'soundfile' unavailable; using librosa fallback.", file=sys.stderr)

# Force only the task (no language token available for CNH)
FORCED_DECODER_IDS = processor.get_decoder_prompt_ids(
    task="transcribe",
    language=None,
    no_timestamps=True
)

# Translator (prefer 'cnh', fallback to auto)
try:
    translator = GoogleTranslator(source="cnh", target="en")
    print("✅ Translator: Hakha Chin (cnh) → en")
except Exception:
    translator = GoogleTranslator(source="auto", target="en")
    print("⚠️ Translator: auto-detect → en", file=sys.stderr)

# ---------- Params ----------
SR_TARGET = 16000
CHUNK_SEC = 30.0
STRIDE_SEC = 2.0
ENERGY_GATE_DB = -60.0     # more permissive for phone mics
NONSILENCE_TOP_DB = 32     # librosa.effects.split(top_db=..)
PEAK_TARGET = 0.95         # peak normalize to ~ -0.45 dBFS

# ---------- Audio utils ----------
def load_audio_16k_mono(filepath: str):
    """Load -> mono -> 16k -> float32."""
    if sf is not None:
        try:
            data, sr = sf.read(filepath, dtype="float32", always_2d=True)
            data = data.mean(axis=1)
            if sr != SR_TARGET:
                data = librosa.resample(y=data, orig_sr=sr, target_sr=SR_TARGET)
            return data.astype(np.float32), SR_TARGET
        except Exception as e:
            print(json.dumps({"loader":"soundfile_failed","err":str(e)}), file=sys.stderr)

    data, sr = librosa.load(filepath, sr=None, mono=False)
    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = data.mean(axis=0)
    if sr != SR_TARGET:
        data = librosa.resample(y=data, orig_sr=sr, target_sr=SR_TARGET)
    return data.astype(np.float32), SR_TARGET

def looks_like_junk(text: str) -> bool:
    if not text: 
        return True
    t = text.lower()
    # too many quotes or repeated tiny syllables
    if t.count('"') >= 4: 
        return True
    # repeated 1-2 char syllables (mi mi mi / pi pi pi)
    if re.search(r'\b(\w{1,2})(\s+\1){4,}\b', t):
        return True
    # very short after stripping punctuation
    core = re.sub(r'[^a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]+', ' ', t).strip()
    if len(core) < 6:
        return True
    return False

def simple_gain_normalize(x: np.ndarray, peak_target=PEAK_TARGET):
    """Peak normalize & mild rms lift to help very quiet captures."""
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x)) + 1e-9)
    if peak < 1e-6:
        return x  # near silence
    x = x * (peak_target / peak)
    # mild rms lift if still too quiet
    rms = float(np.sqrt(np.mean(x**2)) + 1e-9)
    if rms < 0.01:
        x *= 0.01 / rms
    return np.clip(x, -1.0, 1.0)

def rms_db(x: np.ndarray) -> float:
    eps = 1e-10
    rms = float(np.sqrt(np.mean(x**2) + eps))
    return 20.0 * np.log10(rms + eps)

def non_silent_regions(x: np.ndarray, sr: int):
    """Return list of (start, end) sample indices that are non-silent."""
    # librosa's split finds intervals above (max - top_db)
    intervals = librosa.effects.split(x, top_db=NONSILENCE_TOP_DB)
    return [(int(s), int(e)) for s, e in intervals]

def chunk_indices(n_samples: int, sr: int, chunk_sec=CHUNK_SEC, stride_sec=STRIDE_SEC):
    """Yield 30s windows with overlap."""
    win = int(chunk_sec * sr)
    hop = int((chunk_sec - stride_sec) * sr)
    if win >= n_samples:
        yield 0, n_samples
        return
    start = 0
    while start < n_samples:
        end = min(start + win, n_samples)
        yield start, end
        if end == n_samples:
            break
        start += hop

# ---------- Decoding ----------
def decode_chunk(audio_16k: np.ndarray, forced=True) -> tuple[str, float]:
    inputs = processor(audio_16k, sampling_rate=SR_TARGET, return_tensors="pt").input_features.to(device)
    gen_kwargs = dict(
        max_length=320,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
        num_beams=5,
        temperature=0.0,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    if forced:
        gen_kwargs["forced_decoder_ids"] = FORCED_DECODER_IDS

    with torch.no_grad():
        out = model.generate(inputs, **gen_kwargs)

    # Text
    text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
    low = text.lower()
    if low.startswith("(") and "foreign language" in low:
        text = ""

    # ---- Confidence (fixed) ----
    conf = -10.0  # default low
    try:
        # Prefer sequences_scores if present (beam search returns it)
        if hasattr(out, "sequences_scores") and out.sequences_scores is not None:
            conf = float(out.sequences_scores[0].item())
        elif out.scores:
            # Map each time-step score tensor (num_beams, vocab) to the chosen token id
            seq_scores = []
            # For beam search, take the first returned sequence; align steps by position
            chosen_seq = out.sequences[0]  # [seq_len]
            # Skip the BOS token at position 0; out.scores has length = generated steps
            for step_idx, step_scores in enumerate(out.scores, start=1):
                chosen_id = int(chosen_seq[step_idx].item()) if hasattr(chosen_seq[step_idx], "item") else int(chosen_seq[step_idx])
                # pick beam row 0 (the best beam row for the returned sequence)
                beam_row = 0
                log_probs = torch.log_softmax(step_scores[beam_row], dim=-1)  # (vocab_size,)
                seq_scores.append(float(log_probs[chosen_id].item()))
            if seq_scores:
                conf = float(np.mean(seq_scores))
    except Exception as _:
        # leave conf at default if anything odd happens
        pass
    # ----------------------------

    return text, conf

def transcribe_whole(audio: np.ndarray, sr: int) -> str:
    audio = simple_gain_normalize(audio)
    intervals = non_silent_regions(audio, sr)
    if not intervals:
        return ""

    pieces = []
    for s0, e0 in intervals:
        seg = audio[s0:e0]
        if rms_db(seg) < ENERGY_GATE_DB:
            continue
        for s, e in chunk_indices(len(seg), sr):
            chunk = seg[s:e]
            if rms_db(chunk) < ENERGY_GATE_DB:
                continue

            text, conf = decode_chunk(chunk, forced=True)
            if (not text) or looks_like_junk(text) or conf < -1.6:
                # retry unforced once
                text2, conf2 = decode_chunk(chunk, forced=False)
                if (not text2) or looks_like_junk(text2) or conf2 < -1.6:
                    continue
                text, conf = text2, conf2

            pieces.append(text)

    full = " ".join(pieces).strip()
    return full

# ---------- Gradio handlers ----------
def transcribe_audio(audio_file: str) -> str:
    try:
        if not audio_file:
            return "❌ Please upload an audio file!"
        audio, sr = load_audio_16k_mono(audio_file)
        text = transcribe_whole(audio, sr)
        return text or "⚠️ No speech confidently recognized."
    except Exception as e:
        return f"❌ Error: {str(e)}"

def translate_text(text: str) -> str:
    if not text or text.startswith("❌"):
        return ""
    try:
        return translator.translate(text)
    except Exception:
        try:
            return GoogleTranslator(source="auto", target="en").translate(text)
        except Exception as e:
            return f"[Translation unavailable: {e}]"

def process_audio(audio_file: str):
    if not audio_file:
        return "❌ Upload audio!", "", ""
    try:
        transcription = transcribe_audio(audio_file)

        audio, sr = load_audio_16k_mono(audio_file)
        duration = len(audio) / sr
        stats = f"**Duration:** {duration:.2f}s | **Device:** {device.upper()} | **SR:** {sr} Hz | **Chunk:** {CHUNK_SEC:.0f}s | **Gate:** {ENERGY_GATE_DB:.0f} dB"

        translation = translate_text(transcription)
        return transcription, translation, stats
    except Exception as e:
        return f"❌ {str(e)}", "", ""

# ---------------- UI ----------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎤 Hakha Chin Speech-to-Text (V2 - Fixed)\nUpload audio → Hakha Chin transcription → English translation")
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio Input")
            btn = gr.Button("🎯 Transcribe", variant="primary")
        with gr.Column():
            transcription_out = gr.Textbox(label="📝 Hakha Chin", lines=10)
            translation_out = gr.Textbox(label="🌍 English", lines=10)
            stats_out = gr.Markdown()
    btn.click(fn=process_audio, inputs=audio_input, outputs=[transcription_out, translation_out, stats_out])
    audio_input.change(fn=process_audio, inputs=audio_input, outputs=[transcription_out, translation_out, stats_out])

print("\n🚀 Launching...")
demo.launch(share=True, debug=True)