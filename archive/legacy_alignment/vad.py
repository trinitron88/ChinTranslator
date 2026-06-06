from faster_whisper import WhisperModel
import torchaudio, torch, os, tempfile
import string

# ---------------- VAD ----------------
def silero_vad_windows(audio_path, sr_target=16000, min_sil_ms=500, pad_ms=140, merge_gap_ms=350):
    wav, sr = torchaudio.load(audio_path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != sr_target:
        wav = torchaudio.functional.resample(wav, sr, sr_target)
    sr = sr_target
    samples = wav.squeeze(0)

    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    get_speech_timestamps = utils[0]

    ts = get_speech_timestamps(samples, model, sampling_rate=sr,
                               min_silence_duration_ms=min_sil_ms,
                               speech_pad_ms=pad_ms)

    merged=[]
    for t in ts:
        if not merged or (t["start"]-merged[-1]["end"]) > int(merge_gap_ms * sr / 1000):
            merged.append(t.copy())
        else:
            merged[-1]["end"] = t["end"]

    # Debug: how many VAD windows did we get?
    print(f"  Silero VAD windows: {len(merged)}")

    return [(t["start"]/sr, t["end"]/sr) for t in merged]


# ---------------- Garbage check ----------------
def looks_garbage(txt, non_ascii_ratio=0.30):
    if not txt.strip():
        return True
    n_non_ascii = sum(1 for ch in txt if ch not in string.printable)
    return (n_non_ascii / max(1, len(txt))) > non_ascii_ratio


# ---------------- Segment + decode ----------------
def extract_segments(audio_path, model_name="medium"):
    """
    Run Silero VAD over the whole file, then transcribe ONLY the VAD
    windows with faster-whisper. Each VAD window is cut out as its own
    temporary wav so Whisper never sees the surrounding silence / noise.

    Returns a list of segments:
        {"start": float, "end": float, "text": str, "words": [...]} 
    where start/end are absolute seconds in the original file.
    """
    # First, get VAD windows in seconds
    wins = silero_vad_windows(audio_path)

    # Load the full audio once so we can slice by sample index
    wav, sr = torchaudio.load(audio_path)

    # If VAD somehow found nothing, fall back to "whole file" so we still get output
    if not wins:
        print("  ⚠️ Silero VAD found 0 windows; falling back to full-file decode.")
        total_dur = wav.shape[1] / float(sr)
        wins = [(0.0, total_dur)]

    # Whisper model
    model = WhisperModel(model_name, device="cuda", compute_type="float16")

    segments = []

    for s0, s1 in wins:
        # Convert VAD window [s0, s1] → sample indices for this file
        start_sample = max(0, int(s0 * sr))
        end_sample = min(wav.shape[1], int(s1 * sr))

        if end_sample <= start_sample:
            continue

        # Slice this window
        chunk = wav[:, start_sample:end_sample]

        # Write to a temp wav for faster-whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        torchaudio.save(tmp_path, chunk, sr)

        try:
            part, _ = model.transcribe(
                tmp_path,
                task="transcribe",
                condition_on_previous_text=False,
                word_timestamps=True,
                beam_size=5,
                temperature=0.0,
                vad_filter=False,  # we already did VAD with Silero
            )

            # Join text just for the garbage check
            text = "".join(s.text for s in part).strip()

            # If it still looks like garbage, just skip this window
            if looks_garbage(text):
                continue

            # Collect segments with absolute timestamps (offset by s0)
            for s in part:
                segments.append({
                    "start": s.start + s0,
                    "end":   s.end   + s0,
                    "text":  s.text.strip(),
                    "words": [
                        {
                            "start": (w.start + s0),
                            "end":   (w.end   + s0),
                            "word":  w.word,
                        }
                        for w in (s.words or [])
                    ],
                })
        finally:
            # Always clean up the temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    segments.sort(key=lambda x: x["start"])
    return segments


# ---------------- Refinement ----------------
def refine_by_word_gaps(segments, max_gap_s=0.60, max_len_s=20.0):
    # Handle case where VAD/Whisper produced no segments
    if not segments:
        print("  ⚠️ No segments passed into refine_by_word_gaps; returning empty list.")
        return []

    out = []
    cur = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "text": segments[0]["text"],
    }

    for seg in segments[1:]:
        gap = seg["start"] - cur["end"]
        proposed_len = seg["end"] - cur["start"]

        if gap < max_gap_s and proposed_len < max_len_s:
            cur["end"] = seg["end"]
            cur["text"] += " " + seg["text"]
        else:
            out.append(cur)
            cur = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }

    out.append(cur)
    return out

import os, glob, json

AUDIO_DIR = "/content/drive/MyDrive/ChinTranslator/Audio"
OUT_DIR   = "/content/drive/MyDrive/ChinTranslator/Out"
os.makedirs(OUT_DIR, exist_ok=True)

# Only process James chapters, e.g. james_01.mp3, james_02.mp3, etc.
TARGET_PREFIX = "james_"
def save_srt(items, path):
    def ts(t):
        h=int(t//3600); m=int((t%3600)//60); s=int(t%60); ms=int(round((t-int(t))*1000))
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(items, 1):
            f.write(f"{i}\n{ts(seg['start'])} --> {ts(seg['end'])}\n{seg['text'].strip()}\n\n")

def save_json(items, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def transcribe_file(audio_path, model_name="large-v3"):
    """
    Runs VAD → segmentation → stable decode → (optional) English QA join.
    Returns:
        refined_segments (list of dicts)
        english_text (for quick sanity check only)
    """
    print(f"\n🎧 Processing: {audio_path}")

    # Step 1: Segment
    raw = extract_segments(audio_path, model_name=model_name)
    print(f"  Raw segments:     {len(raw)}")

    if not raw:
        print("  ⚠️ No segments from VAD/Whisper; skipping refinement for this file.")
        return [], ""

    # Step 2: Refine small gaps
    refined = refine_by_word_gaps(raw, max_gap_s=0.60, max_len_s=20.0)
    print(f"  Refined segments: {len(refined)}")

    # Quick joined text for sanity check (not used as translation!)
    english_joined = " ".join([seg["text"] for seg in refined]).strip()

    return refined, english_joined

files = sorted(
    f
    for f in glob.glob(os.path.join(AUDIO_DIR, "*.mp3")) +
             glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    if os.path.basename(f).startswith(TARGET_PREFIX)
)

print(f"Found {len(files)} James files\n")

for i, path in enumerate(files, 1):
    base = os.path.splitext(os.path.basename(path))[0]
    print(f"[{i}/{len(files)}] {base}")

    # ⚠️ T4 users: use model_name="medium"
    refined, english = transcribe_file(path, model_name="medium")

    save_srt(refined, os.path.join(OUT_DIR, base + ".srt"))
    save_json(refined, os.path.join(OUT_DIR, base + ".json"))

print("\n✅ Batch done →", OUT_DIR)