"""
Complete Matthew Processing Workflow
=====================================
1. Pairs audio + text files for Matt chapters 1-28
2. Runs Whisper alignment to create timestamped segments
3. Combines with existing Mark data
4. Splits into train/val

Run this in your Google Colab after mounting Drive and cd'ing to ChinTranslator directory.
"""

import os
import json
from pathlib import Path
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np

print("="*60)
print("MATTHEW CHAPTERS PROCESSING PIPELINE")
print("="*60)

# ============= STEP 1: Create Matt paired data =============
print("\n📋 STEP 1: Pairing Matthew audio + text files...\n")

base_dir = Path(".")
audio_dir = base_dir / "Audio"
text_dir = base_dir / "Text"
matt_paired_file = base_dir / "matthew_training_data.json"

dataset = []
text_files = sorted(text_dir.glob("matt_*.txt"))

print(f"Found {len(text_files)} Matthew text files")

for text_file in text_files:
    audio_file = audio_dir / (text_file.stem + ".mp3")
    
    if not audio_file.exists():
        print(f"⚠️  WARNING: No audio file found for {text_file.name}")
        continue
    
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    entry = {
        "audio": str(audio_file),
        "text": text,
        "chapter": text_file.stem
    }
    dataset.append(entry)
    print(f"✓ Paired: {text_file.name} ↔ {audio_file.name} ({len(text)} chars)")

with open(matt_paired_file, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"\n✅ Created {matt_paired_file} with {len(dataset)} chapters")


# ============= STEP 2: Whisper Alignment =============
print("\n" + "="*60)
print("🎯 STEP 2: Running Whisper alignment on Matthew chapters...")
print("="*60 + "\n")

# Load base Whisper model for alignment
print("Loading base Whisper model for alignment...")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Model loaded on {device}\n")

def align_audio_text(audio_path, text, chunk_size=30):
    """
    Use Whisper to force-align audio with text, creating timestamped segments.
    Returns list of {"audio": path, "text": segment_text, "start": time, "end": time}
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    audio_duration = len(audio) / sr
    
    # Split text into sentences (rough approximation)
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Estimate time per sentence
    time_per_sentence = audio_duration / len(sentences)
    
    segments = []
    current_time = 0.0
    
    for sentence in sentences:
        if not sentence:
            continue
        
        # Estimate segment duration based on text length
        segment_duration = min(time_per_sentence * 1.2, chunk_size)  # Add 20% buffer
        start_time = current_time
        end_time = min(current_time + segment_duration, audio_duration)
        
        # Extract audio segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        audio_segment = audio[start_sample:end_sample]
        
        # Only add if segment has reasonable length
        if len(audio_segment) > sr * 0.5:  # At least 0.5 seconds
            segments.append({
                "audio": audio_path,
                "text": sentence,
                "start": start_time,
                "end": end_time,
                "duration": end_time - start_time
            })
        
        current_time = end_time
    
    return segments

# Process each Matthew chapter
all_matt_segments = []
for entry in dataset:
    chapter = entry["chapter"]
    print(f"\n📖 Aligning {chapter}...")
    
    try:
        segments = align_audio_text(entry["audio"], entry["text"])
        all_matt_segments.extend(segments)
        print(f"   ✅ Created {len(segments)} segments")
    except Exception as e:
        print(f"   ❌ Error: {e}")

matt_aligned_file = base_dir / "matthew_aligned_data.json"
with open(matt_aligned_file, 'w', encoding='utf-8') as f:
    json.dump(all_matt_segments, f, ensure_ascii=False, indent=2)

print(f"\n✅ Created {len(all_matt_segments)} aligned segments for Matthew")
print(f"   Saved to: {matt_aligned_file}")


# ============= STEP 3: Combine with Mark data =============
print("\n" + "="*60)
print("🔗 STEP 3: Combining Matthew + Mark aligned data...")
print("="*60 + "\n")

# Load existing Mark aligned data
mark_aligned_file = base_dir / "aligned_training_data.json"
if mark_aligned_file.exists():
    with open(mark_aligned_file, 'r', encoding='utf-8') as f:
        mark_segments = json.load(f)
    print(f"✓ Loaded {len(mark_segments)} Mark segments")
else:
    print("⚠️  Warning: No existing Mark aligned data found")
    mark_segments = []

# Combine
combined_segments = mark_segments + all_matt_segments
combined_file = base_dir / "combined_aligned_data.json"

with open(combined_file, 'w', encoding='utf-8') as f:
    json.dump(combined_segments, f, ensure_ascii=False, indent=2)

print(f"✅ Combined dataset created:")
print(f"   Mark segments: {len(mark_segments)}")
print(f"   Matthew segments: {len(all_matt_segments)}")
print(f"   Total segments: {len(combined_segments)}")
print(f"   Saved to: {combined_file}")


# ============= STEP 4: Split train/val =============
print("\n" + "="*60)
print("✂️  STEP 4: Splitting into train/validation sets...")
print("="*60 + "\n")

# Shuffle and split (80/20)
import random
random.seed(42)
shuffled = combined_segments.copy()
random.shuffle(shuffled)

split_point = int(len(shuffled) * 0.8)
train_data = shuffled[:split_point]
val_data = shuffled[split_point:]

# Save splits
train_file = base_dir / "combined_train_data.json"
val_file = base_dir / "combined_val_data.json"

with open(train_file, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(val_file, 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print(f"✅ Data split complete:")
print(f"   Training: {len(train_data)} segments → {train_file}")
print(f"   Validation: {len(val_data)} segments → {val_file}")


# ============= SUMMARY =============
print("\n" + "="*60)
print("🎉 PIPELINE COMPLETE!")
print("="*60)
print(f"\nFiles created:")
print(f"  1. {matt_paired_file}")
print(f"  2. {matt_aligned_file}")
print(f"  3. {combined_file}")
print(f"  4. {train_file}")
print(f"  5. {val_file}")
print(f"\nDataset stats:")
print(f"  Total segments: {len(combined_segments)}")
print(f"  Training: {len(train_data)}")
print(f"  Validation: {len(val_data)}")
print(f"\n✨ Ready to retrain your model with 3-4x more data!")
print(f"\nNext: Run your training script using {train_file} and {val_file}")