#!/usr/bin/env python3
"""
Fixed-Duration Alignment Script
================================
Simple approach: Split audio into fixed 10-second chunks,
divide text proportionally based on chapter-level char/second rate.

This won't be perfect, but it will be CONSISTENT and won't create
the wild misalignments that Whisper speech detection caused.
"""

import json
import librosa
from pathlib import Path
import random

print("="*60)
print("FIXED-DURATION ALIGNMENT (10-second chunks)")
print("="*60)

# Configuration
CHUNK_DURATION = 10.0  # seconds
OVERLAP = 0.5  # seconds of overlap between chunks

BOOKS = [
    ("Matthew", "matt"),
    ("Mark", "mark"),
    ("James", "james")
]

base_dir = Path(".")
audio_dir = base_dir / "Audio"
text_dir = base_dir / "Text"

def split_text_proportionally(text, num_chunks):
    """
    Split text into num_chunks pieces, trying to break on sentence boundaries.
    """
    # Split into sentences
    import re
    sentences = re.split(r'(?<=[.!?;…])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    # Calculate roughly how many sentences per chunk
    sentences_per_chunk = len(sentences) / num_chunks
    
    chunks = []
    current_chunk = []
    sentence_count = 0
    
    for sentence in sentences:
        current_chunk.append(sentence)
        sentence_count += 1
        
        # When we've accumulated enough sentences for a chunk
        if sentence_count >= sentences_per_chunk and len(chunks) < num_chunks - 1:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            sentence_count = 0
    
    # Add remaining sentences to last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # If we have fewer chunks than expected, pad with empty strings
    while len(chunks) < num_chunks:
        chunks.append("")
    
    return chunks

def align_chapter_fixed_duration(audio_path, text, chunk_duration=10.0, overlap=0.5):
    """
    Align chapter by splitting into fixed-duration audio chunks
    and proportionally dividing text.
    """
    # Load audio to get duration
    audio, sr = librosa.load(audio_path, sr=16000)
    total_duration = len(audio) / sr
    
    # Calculate number of chunks (with overlap)
    step_size = chunk_duration - overlap
    num_chunks = int((total_duration - overlap) / step_size) + 1
    
    print(f"  Audio duration: {total_duration:.1f}s")
    print(f"  Creating {num_chunks} chunks of {chunk_duration}s (overlap: {overlap}s)")
    
    # Split text proportionally
    text_chunks = split_text_proportionally(text, num_chunks)
    
    # Create segments
    segments = []
    for i in range(num_chunks):
        start_time = i * step_size
        end_time = min(start_time + chunk_duration, total_duration)
        
        # Skip if chunk is too short
        if end_time - start_time < 2.0:
            continue
        
        # Skip if no text for this chunk
        if not text_chunks[i] or len(text_chunks[i].strip()) < 5:
            continue
        
        segments.append({
            "audio": str(audio_path),
            "start": start_time,
            "end": end_time,
            "text": text_chunks[i],
            "duration": end_time - start_time
        })
    
    return segments

# Process all books
all_segments = []
book_stats = {}

for book_name, file_prefix in BOOKS:
    print(f"\n{'='*60}")
    print(f"Processing {book_name}")
    print('='*60)
    
    text_files = sorted(text_dir.glob(f"{file_prefix}_*.txt"))
    
    if not text_files:
        print(f"⚠️  No text files found for {book_name}")
        continue
    
    print(f"Found {len(text_files)} chapters")
    book_segments = []
    
    for text_file in text_files:
        chapter_name = text_file.stem
        audio_file = audio_dir / (text_file.stem + ".mp3")
        
        if not audio_file.exists():
            print(f"\n⚠️  No audio for {chapter_name}")
            continue
        
        print(f"\n📖 {chapter_name}")
        
        # Load text
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        print(f"  Text length: {len(text)} characters")
        
        try:
            # Align with fixed duration
            segments = align_chapter_fixed_duration(
                str(audio_file),
                text,
                chunk_duration=CHUNK_DURATION,
                overlap=OVERLAP
            )
            
            # Add metadata
            for seg in segments:
                seg["book"] = book_name
                seg["chapter"] = chapter_name
            
            book_segments.extend(segments)
            all_segments.extend(segments)
            
            print(f"  ✅ Created {len(segments)} segments")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    book_stats[book_name] = len(book_segments)
    print(f"\n✅ {book_name} total: {len(book_segments)} segments")
    
    # Save individual book file
    book_file = base_dir / f"{file_prefix}_aligned_fixed.json"
    with open(book_file, 'w', encoding='utf-8') as f:
        json.dump(book_segments, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved to {book_file}")

# Summary
print("\n" + "="*60)
print("ALIGNMENT COMPLETE")
print("="*60)
print(f"\nTotal segments: {len(all_segments)}")
print("\nSegments per book:")
for book, count in book_stats.items():
    percentage = (count / len(all_segments)) * 100 if all_segments else 0
    print(f"  {book:12s}: {count:4d} ({percentage:5.1f}%)")

# Save combined file
combined_file = base_dir / "all_books_aligned_fixed.json"
with open(combined_file, 'w', encoding='utf-8') as f:
    json.dump(all_segments, f, ensure_ascii=False, indent=2)
print(f"\n💾 Combined data saved to {combined_file}")

# Split into train/val
print("\n✂️  Splitting into train/validation (80/20)...")
random.seed(42)
shuffled = all_segments.copy()
random.shuffle(shuffled)

split_point = int(len(shuffled) * 0.8)
train_data = shuffled[:split_point]
val_data = shuffled[split_point:]

# Save splits
train_file = base_dir / "train_data_fixed.json"
val_file = base_dir / "val_data_fixed.json"

with open(train_file, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(val_file, 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print(f"✅ Train: {len(train_data)} segments → {train_file}")
print(f"✅ Val:   {len(val_data)} segments → {val_file}")

print("\n" + "="*60)
print("🎉 Done! Fixed-duration alignment complete!")
print("="*60)
print("\nNext steps:")
print("1. Manually verify a few segments are better aligned")
print("2. Update training script to use train_data_fixed.json")
print("3. Retrain and test!")
print("\n⚠️  NOTE: This alignment is rougher than Whisper-based,")
print("   but should be more CONSISTENT. The key is that text")
print("   and audio boundaries should actually correspond now.")