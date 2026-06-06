#!/usr/bin/env python3

import whisper
import json
import re
import random
from pathlib import Path
from statistics import median

print("="*60)
print("Creating Aligned Training Data - ALL CHAPTERS")
print("Mark (16) + Matthew (28) = 44 total")
print("="*60)

# ----------------------------------------------------
# Load Whisper Model (use "base" when you care about accuracy)
# ----------------------------------------------------
print("\n📥 Loading Whisper model...")
model = whisper.load_model("tiny")  # swap to "base" later for cleaner segmentation
print("✅ Model loaded")

# ----------------------------------------------------
# Load Combined Chapter Data
# ----------------------------------------------------
print("\n📂 Loading paired data...")
all_chapters = []

for filename in ["training_data.json", "matthew_training_data.json"]:
    file_path = Path(filename)
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_chapters.extend(data)
        print(f"✓ Loaded {len(data)} from {filename}")
    else:
        print(f"⚠️  {filename} not found")

if not all_chapters:
    print("❌ ERROR: No chapter data found")
    exit(1)

print(f"\n📊 Total chapters: {len(all_chapters)}")

# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
def split_sentences(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r'(?<=[\.!\?;…])\s+(?=[A-Z0-9“"])', text)
    return [p.strip().strip('"“”') for p in parts if p.strip()]

def chars_per_second(seg_text, seg_dur, default_cps=15.0):
    c = max(len(seg_text.strip()), 1)
    d = max(seg_dur, 0.1)
    est = c / d
    return max(min(est, 40.0), 6.0) if c > 5 else default_cps

# ----------------------------------------------------
# Alignment
# ----------------------------------------------------
aligned_segments = []
failed = []

for idx, chapter in enumerate(all_chapters, 1):
    chapter_name = chapter["chapter"]
    full_text = chapter["text"]
    audio_path = chapter["audio"]

    print(f"\n[{idx}/{len(all_chapters)}] 🔄 Processing {chapter_name}...")

    try:
        result = model.transcribe(
            audio_path,
            task="transcribe",
            verbose=False,
            word_timestamps=True,
            condition_on_previous_text=False,
            temperature=0.0,
            beam_size=5,
            best_of=5
        )

        whisper_segments = result["segments"]
        print(f"    Whisper segments: {len(whisper_segments)}")

        sentences = split_sentences(full_text)
        print(f"    Text sentences: {len(sentences)}")

        if not whisper_segments or not sentences:
            failed.append(chapter_name)
            continue

        # Compute global cps for fallback
        cps_values = [
            chars_per_second(seg.get("text", ""), seg["end"] - seg["start"])
            for seg in whisper_segments
        ]
        global_cps = median(cps_values) if cps_values else 15.0

        sidx = 0
        created = 0

        for seg in whisper_segments:
            seg_start, seg_end = seg["start"], seg["end"]
            seg_dur = max(seg_end - seg_start, 0.1)

            target_cps = chars_per_second(seg.get("text", ""), seg_dur, global_cps)
            char_budget = int(target_cps * seg_dur)

            picked = []
            total_chars = 0

            while sidx < len(sentences):
                next_len = len(sentences[sidx])
                if not picked or (total_chars + next_len <= int(char_budget * 1.15)):
                    picked.append(sentences[sidx])
                    total_chars += next_len
                    sidx += 1
                else:
                    break

            if not picked and sidx < len(sentences):
                picked = [sentences[sidx]]
                sidx += 1

            if picked:
                pad = 0.25
                aligned_segments.append({
                    "audio": audio_path,
                    "start": max(seg_start - pad, 0),
                    "end": seg_end + pad,
                    "text": " ".join(picked),
                    "chapter": chapter_name
                })
                created += 1

            if sidx >= len(sentences):
                break

        print(f"    ✅ Created {created} aligned segments")

    except Exception as e:
        print(f"    ❌ Error: {e}")
        failed.append(chapter_name)

print("\n" + "="*60)
print("Alignment complete")
print("="*60)
print(f"Total aligned segments: {len(aligned_segments)}")
print(f"Failed chapters: {failed}\n")

# ----------------------------------------------------
# Save Full Dataset
# ----------------------------------------------------
with open("all_chapters_aligned_data.json", "w", encoding="utf-8") as f:
    json.dump(aligned_segments, f, ensure_ascii=False, indent=2)
print("💾 Saved all segments → all_chapters_aligned_data.json")

# ----------------------------------------------------
# Train / Val Split
# ----------------------------------------------------
random.seed(42)
random.shuffle(aligned_segments)
split = int(len(aligned_segments) * 0.8)
train = aligned_segments[:split]
val = aligned_segments[split:]

with open("combined_train_data.json", "w", encoding="utf-8") as f:
    json.dump(train, f, ensure_ascii=False, indent=2)

with open("combined_val_data.json", "w", encoding="utf-8") as f:
    json.dump(val, f, ensure_ascii=False, indent=2)

print(f"✅ Train: {len(train)} → combined_train_data.json")
print(f"✅ Val:   {len(val)} → combined_val_data.json")
print("\n🎉 Done.")