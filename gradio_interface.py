"""
Gradio Interface - FIXED VERSION (Option A)
- Normalizes mic/uploads to 16 kHz mono
- Forces Hakha Chin transcription (no English fallback)
- Always attempts translation to English (with resilient fallback)
"""

import json
import sys
import numpy as np
import gradio as gr
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from deep_translator import GoogleTranslator

print("🤖 Loading Hakha Chin Whisper V2...")
MODEL_PATH = "./whisper-hakha-chin-v2"

processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Model loaded on {device}")

# Try to import soundfile for robust loading; fall back to librosa if missing.
try:
    import soundfile as sf
except Exception:
    sf = None
    print("ℹ️ 'soundfile' not available, will fall back to librosa loader.", file=sys.stderr)

# Prepare forced decoder IDs ONCE (forces Hakha Chin + transcribe)
FORCED_DECODER_IDS = processor.get_decoder_prompt_ids(language="cnh", task="transcribe")

# Initialize translator (prefer explicit CNH, fall back to auto)
try:
    translator = GoogleTranslator(source="cnh", target="en")
    print("✅ Translator: Hakha Chin (cnh) → en")
except Exception:
    translator = GoogleTranslator(source="auto", target="en")
    print("⚠️ Translator: auto-detect → en", file=sys.stderr)


def load_audio_16k_mono(filepath: str):
    """
    Robustly load audio, convert to 16 kHz mono float32 for Whisper.
    Works for mic temp files (m4a/webm/wav) and uploads.
    """
    # First try soundfile (fast & precise), then librosa as a universal fallback.
    if sf is not None:
        try:
            data, sr = sf.read(filepath, dtype="float32", always_2d=True)  # (n_samples, n_channels)
            data = data.mean(axis=1)  # mono
            if sr != 16000:
                data = librosa.resample(y=data, orig_sr=sr, target_sr=16000)
                sr = 16000
            return data, sr
        except Exception as e:
            print(json.dumps({"loader": "soundfile_failed", "err": str(e)}), file=sys.stderr)

    # Fallback: librosa (handles many containers/codecs via audioread/ffmpeg)
    data, sr = librosa.load(filepath, sr=None, mono=False)
    if isinstance(data, np.ndarray) and data.ndim == 2:
        data = data.mean(axis=0)
    if sr != 16000:
        data = librosa.resample(y=data, orig_sr=sr, target_sr=16000)
        sr = 16000
    return data.astype(np.float32), sr


def transcribe_audio(audio_file: str) -> str:
    """Transcribe with forced Hakha Chin settings to prevent English fallback."""
    try:
        if not audio_file:
            return "❌ Please upload an audio file!"

        audio, sr = load_audio_16k_mono(audio_file)

        input_features = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=FORCED_DECODER_IDS,  # KEY: force CNH + transcribe
                max_length=448,               # Whisper's comfortable upper bound
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
                return_dict_in_generate=False
            )

        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0].strip()

        # Defensive: if a fallback string slipped in from tokens seen during finetune
        low = transcription.lower()
        if low.startswith("(") and "foreign language" in low:
            transcription = ""

        return transcription or "⚠️ No speech confidently recognized."
    except Exception as e:
        return f"❌ Error: {str(e)}"


def translate_text(text: str) -> str:
    """Translate to English (always attempt; resilient fallback)."""
    if not text or text.startswith("❌"):
        return ""
    try:
        return translator.translate(text)  # prefers source='cnh'
    except Exception:
        try:
            return GoogleTranslator(source="auto", target="en").translate(text)
        except Exception as e:
            return f"[Translation unavailable: {e}]"


def process_audio(audio_file: str):
    """Full pipeline: load → transcribe → translate → stats."""
    if not audio_file:
        return "❌ Upload audio!", "", ""

    try:
        # Transcribe
        transcription = transcribe_audio(audio_file)

        # Stats based on the *processed* signal fed to the model
        audio, sr = load_audio_16k_mono(audio_file)
        duration = len(audio) / sr
        stats = f"**Duration:** {duration:.2f}s | **Device:** {device.upper()} | **SR:** {sr} Hz"

        # Always attempt translation (Option A)
        translation = translate_text(transcription)

        return transcription, translation, stats

    except Exception as e:
        return f"❌ {str(e)}", "", ""


# ---------------- UI ----------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎤 Hakha Chin Speech-to-Text (V2 - Fixed)
        Upload audio → Hakha Chin transcription → English translation
        """
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio Input"
            )
            btn = gr.Button("🎯 Transcribe", variant="primary")

        with gr.Column():
            transcription_out = gr.Textbox(
                label="📝 Hakha Chin",
                lines=8
            )
            translation_out = gr.Textbox(
                label="🌍 English",
                lines=8
            )
            stats_out = gr.Markdown()

    # Events
    btn.click(
        fn=process_audio,
        inputs=audio_input,
        outputs=[transcription_out, translation_out, stats_out]
    )

    audio_input.change(
        fn=process_audio,
        inputs=audio_input,
        outputs=[transcription_out, translation_out, stats_out]
    )

print("\n🚀 Launching...")
demo.launch(share=True, debug=True)