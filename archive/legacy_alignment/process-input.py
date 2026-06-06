"""
Complete Multi-Book Processing Workflow
========================================
Processes Matthew, Mark, Acts, and James:
1. Pairs audio + text files for each book
2. Runs Whisper alignment to create timestamped segments
3. Combines all books together
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
print("MULTI-BOOK PROCESSING PIPELINE")
print("="*60)

# Define the books we're processing
BOOKS = ["Matthew", "Mark", "Acts", "James"]

base_dir = Path(".")
audio_dir = base_dir / "Audio"
text_dir = base_dir / "Text"

# ============= STEP 1: Create paired data for all books =============
print("\n📋 STEP 1: Pairing audio + text files for all books...\n")

all_datasets = {}

for book in BOOKS:
    print(f"\n--- Processing {book} ---")
    dataset = []
    text_files = sorted(text_dir.glob(f"{book}_*.txt"))
    
    print(f"Found {len(text_files)} {book} text files")
    
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
            "chapter": text_file.stem,
            "book": book
        }
        dataset.append(entry)
        print(f"✓ Paired: {text_file.name} → {audio_file.name} ({len(text)} chars)")
    
    all_datasets[book] = dataset
    
    # Save individual book paired data
    book_paired_file = base_dir / f"{book.lower()}_training_data.json"
    with open(book_paired_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Created {book_paired_file} with {len(dataset)} chapters")

total_chapters = sum(len(ds) for ds in all_datasets.values())
print(f"\n✅ Total chapters across all books: {total_chapters}")


# ============= STEP 2: Whisper Alignment =============
print("\n" + "="*60)
print("🎯 STEP 2: Running Whisper alignment on all chapters...")
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

# Process each book
all_aligned_segments = []

for book in BOOKS:
    print(f"\n{'='*60}")
    print(f"Aligning {book}")
    print('='*60)
    
    dataset = all_datasets[book]
    book_segments = []
    
    for entry in dataset:
        chapter = entry["chapter"]
        print(f"\n📖 Aligning {chapter}...")
        
        try:
            segments = align_audio_text(entry["audio"], entry["text"])
            # Add book info to each segment
            for seg in segments:
                seg["book"] = book
            book_segments.extend(segments)
            all_aligned_segments.extend(segments)
            print(f"   ✅ Created {len(segments)} segments")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Save individual book aligned data
    book_aligned_file = base_dir / f"{book.lower()}_aligned_data.json"
    with open(book_aligned_file, 'w', encoding='utf-8') as f:
        json.dump(book_segments, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ {book}: {len(book_segments)} aligned segments")
    print(f"   Saved to: {book_aligned_file}")

# Save combined aligned data
combined_aligned_file = base_dir / "all_books_aligned_data.json"
with open(combined_aligned_file, 'w', encoding='utf-8') as f:
    json.dump(all_aligned_segments, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"✅ Total aligned segments across all books: {len(all_aligned_segments)}")
print(f"   Saved to: {combined_aligned_file}")


# ============= STEP 3: Split train/val =============
print("\n" + "="*60)
print("✂️  STEP 3: Splitting into train/validation sets...")
print("="*60 + "\n")

# Shuffle and split (80/20)
import random
random.seed(42)
shuffled = all_aligned_segments.copy()
random.shuffle(shuffled)

split_point = int(len(shuffled) * 0.8)
train_data = shuffled[:split_point]
val_data = shuffled[split_point:]

# Save splits
train_file = base_dir / "train_data.json"
val_file = base_dir / "val_data.json"

with open(train_file, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(val_file, 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print(f"✅ Data split complete:")
print(f"   Training: {len(train_data)} segments → {train_file}")
print(f"   Validation: {len(val_data)} segments → {val_file}")

# Print book distribution in training set
print(f"\n📊 Book distribution in training set:")
book_counts = {}
for item in train_data:
    book = item.get("book", "Unknown")
    book_counts[book] = book_counts.get(book, 0) + 1

for book in BOOKS:
    count = book_counts.get(book, 0)
    percentage = (count / len(train_data)) * 100
    print(f"   {book}: {count} segments ({percentage:.1f}%)")


# ============= SUMMARY =============
print("\n" + "="*60)
print("🎉 PIPELINE COMPLETE!")
print("="*60)
print(f"\nBooks processed: {', '.join(BOOKS)}")
print(f"\nDataset stats:")
print(f"  Total segments: {len(all_aligned_segments)}")
print(f"  Training: {len(train_data)} ({(len(train_data)/len(all_aligned_segments)*100):.1f}%)")
print(f"  Validation: {len(val_data)} ({(len(val_data)/len(all_aligned_segments)*100):.1f}%)")
print(f"\n✨ Ready to retrain your model!")
print(f"\nNext: Run your training script using {train_file} and {val_file}")