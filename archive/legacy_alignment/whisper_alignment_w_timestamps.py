#!/usr/bin/env python3
"""
Improved Alignment Using Whisper's Speech Detection
====================================================
Uses Whisper to detect where speech actually occurs, then maps
Chin text to those detected segments.

Even though Whisper can't transcribe Chin, it CAN detect:
- Where speech starts and stops
- Natural pauses between utterances
- Prosody boundaries

We'll use these speech boundaries to create better-aligned segments.
"""

import whisper
import json
import re
from pathlib import Path
from typing import List, Dict
import random
from difflib import SequenceMatcher

DEBUG = True

print("="*60)
print("WHISPER-BASED ALIGNMENT FOR CHIN AUDIO")
print("="*60)

# Configuration
BOOKS = [
    ("Matthew", "matt"),
    #("Mark", "mark"),
    #("James", "james")
]

base_dir = Path(".")
audio_dir = base_dir / "Audio"
text_dir = base_dir / "Text"

# Load Whisper for speech detection
print("\n📥 Loading Whisper model for speech detection...")
model = whisper.load_model("base")  # Use base for speed
print("✅ Model loaded\n")



def chunk_by_chars(text: str, target: int = 140) -> List[str]:
    """Fallback chunker: splits text into ~target-char chunks when punctuation is sparse."""
    out, buf = [], []
    count = 0
    for token in re.sub(r"\s+", " ", text).strip().split():
        buf.append(token)
        count += len(token) + 1
        if count >= target:
            out.append(" ".join(buf).strip())
            buf, count = [], 0
    if buf:
        out.append(" ".join(buf).strip())
    return out

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, with a fallback to fixed-size chunks if punctuation is sparse."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Primary: punctuation boundaries
    sentences = re.split(r"(?<=[.!?;…])\s+", text)
    sentences = [s.strip() for s in sentences if s and s.strip()]

    # Fallback: if too few sentences for alignment, chunk by length
    if len(sentences) < 20 and len(text) > 600:
        sentences = chunk_by_chars(text, target=140)

    return sentences


def is_bad_segment(seg: Dict) -> bool:
    """Permissive filter: rely only on timing/energy-like stats.
    We ignore the decoded text, since we're only using Whisper as a VAD.
    """
    # Drop if Whisper thinks it's very likely non-speech.
    if seg.get("no_speech_prob", 0) > 0.85:
        return True
    # Very low confidence overall.
    if seg.get("avg_logprob", 0) < -1.2:
        return True
    # Extremely compressed gibberish.
    if seg.get("compression_ratio", 0) and seg["compression_ratio"] > 2.6:
        return True
    # Otherwise keep; text content is irrelevant for alignment.
    return False


def near_duplicate(a: str, b: str, threshold: float = 0.8) -> bool:
    """Simple similarity check to de-dup repeated segment texts."""
    if not a or not b:
        return False
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio() >= threshold


def align_text_to_speech_segments(
    audio_path: str,
    chin_text: str,
    min_segment_length: float = 1.0,
    max_segment_length: float = 14.0
) -> List[Dict]:
    """
    Use Whisper to detect speech segments, then map Chin text to them.
    
    Args:
        audio_path: Path to audio file
        chin_text: The Chin text to align
        min_segment_length: Minimum segment duration in seconds
        max_segment_length: Maximum segment duration in seconds
    
    Returns:
        List of aligned segments with audio path, start, end, and text
    """
    print(f"  🎧 Detecting speech in audio...")
    
    # Run Whisper to get speech segments (we ignore the transcribed text)
    result = model.transcribe(
        audio_path,
        task="transcribe",                   # do NOT translate; stay in-source language
        verbose=False,
        word_timestamps=True,
        condition_on_previous_text=False,    # prevent error snowball
        temperature=0.0,                     # deterministic
        suppress_blank=True,
        compression_ratio_threshold=2.4,     # drop over-compressed (gibberish) segments
        logprob_threshold=-1.0,              # drop very low-confidence decodes
        no_speech_threshold=0.6              # drop likely-non-speech regions
    )
    if DEBUG:
        lang = result.get("language", None)
        if lang:
            print(f"  🌐 Detected language: {lang}")
    whisper_segments = result["segments"]
    if DEBUG:
        print(f"  🔎 Raw segments preview (up to 5):")
        for _i, _s in enumerate(whisper_segments[:5]):
            print(f"    #{_i}: start={_s.get('start')}, end={_s.get('end')}, "
                  f"dur={(_s.get('end',0)-_s.get('start',0)):.2f}, "
                  f"no_speech_prob={_s.get('no_speech_prob')}, "
                  f"avg_logprob={_s.get('avg_logprob')}, "
                  f"compression_ratio={_s.get('compression_ratio')}, "
                  f"text={repr((_s.get('text') or '').strip())[:60]}")
    
    print(f"  ✓ Detected {len(whisper_segments)} speech segments")
    
    # Quality filter & de-dup Whisper segments before mapping text
    cleaned_segments = []
    last_txt = ""
    for s in whisper_segments:
        if is_bad_segment(s):
            continue
        # keep only segments with valid timing
        if "start" not in s or "end" not in s:
            continue
        dur = s["end"] - s["start"]
        if dur <= 0:
            continue
        # de-dup based on near-identical timing (sometimes Whisper splits/merges oddly)
        if cleaned_segments:
            prev = cleaned_segments[-1]
            if abs(s["start"] - prev["start"]) < 0.05 and abs(s["end"] - prev["end"]) < 0.05:
                continue
        cleaned_segments.append(s)
    whisper_segments = cleaned_segments
    print(f"  ✓ Kept {len(whisper_segments)} clean segments after quality filters")
    if DEBUG and whisper_segments:
        durs = [round(s["end"] - s["start"], 2) for s in whisper_segments if "start" in s and "end" in s]
        if durs:
            print(f"  ⏱️  Durations (first 10): {durs[:10]}")
    # If we were too strict, fall back to timing-only filtering.
    if not whisper_segments:
        if DEBUG:
            print("  ⚠️  No segments after filters; falling back to timing-only VAD.")
        timing_only = []
        for s in result.get("segments", []):
            if "start" not in s or "end" not in s:
                continue
            dur = s["end"] - s["start"]
            if dur <= 0:
                continue
            timing_only.append(s)
        whisper_segments = timing_only
        print(f"  ✓ Fallback kept {len(whisper_segments)} segments (timing-only)")
    
    # Split Chin text into sentences
    chin_sentences = split_into_sentences(chin_text)
    print(f"  ✓ Split text into {len(chin_sentences)} sentences")
    if not chin_sentences and chin_text.strip():
        chin_sentences = [chin_text.strip()]
        if DEBUG:
            print("  🧷 No sentence boundaries found; using whole text as one sentence.")
    
    if not whisper_segments or not chin_sentences:
        print("  ⚠️  No segments or no text - skipping")
        return []
    
    # Calculate total text length for proportional distribution
    total_chars = sum(len(s) for s in chin_sentences)
    
    aligned_segments = []
    sentence_idx = 0
    
    for seg in whisper_segments:
        # Skip segments missing keys (already filtered above, but double-guard)
        if "start" not in seg or "end" not in seg:
            continue
        
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_duration = seg_end - seg_start
        
        # Enforce duration bounds and split long segments
        if seg_duration < max(0.3, min_segment_length):
            continue
        if seg_duration > max_segment_length:
            num_splits = int(seg_duration // max_segment_length) + 1
            split_duration = seg_duration / max(num_splits, 1)
            for i in range(num_splits):
                split_start = seg_start + (i * split_duration)
                split_end = min(seg_end, split_start + split_duration)
                if sentence_idx < len(chin_sentences):
                    aligned_segments.append({
                        "audio": str(audio_path),
                        "start": split_start,
                        "end": split_end,
                        "text": chin_sentences[sentence_idx],
                        "duration": split_end - split_start
                    })
                    sentence_idx += 1
            continue
        
        # Estimate how many sentences fit in this segment based on duration
        # Assume roughly 15 characters per second of speech
        chars_in_segment = int(seg_duration * 15)
        
        # Collect sentences that fit in this segment
        segment_sentences = []
        segment_chars = 0
        
        while sentence_idx < len(chin_sentences):
            next_sentence = chin_sentences[sentence_idx]
            next_len = len(next_sentence)
            
            # Add sentence if we haven't exceeded the char budget (with 20% tolerance)
            if not segment_sentences or segment_chars + next_len <= chars_in_segment * 1.2:
                segment_sentences.append(next_sentence)
                segment_chars += next_len
                sentence_idx += 1
            else:
                break
        
        # If we didn't collect any sentences, force at least one
        if not segment_sentences and sentence_idx < len(chin_sentences):
            segment_sentences.append(chin_sentences[sentence_idx])
            sentence_idx += 1
        
        if segment_sentences:
            aligned_segments.append({
                "audio": str(audio_path),
                "start": seg_start,
                "end": seg_end,
                "text": " ".join(segment_sentences),
                "duration": seg_duration
            })
    
    print(f"  ✅ Created {len(aligned_segments)} aligned segments")
    print(f"  📊 Used {sentence_idx}/{len(chin_sentences)} sentences")
    
    return aligned_segments


# Main processing loop
all_segments = []
book_stats = {}

for book_name, file_prefix in BOOKS:
    print(f"\n{'='*60}")
    print(f"Processing {book_name}")
    print('='*60)
    
    # Find all text files for this book
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
        
        # Load Chin text
        with open(text_file, 'r', encoding='utf-8') as f:
            chin_text = f.read().strip()
        
        print(f"  📝 Text length: {len(chin_text)} characters")
        
        try:
            # Align using Whisper's speech detection
            segments = align_text_to_speech_segments(
                str(audio_file),
                chin_text
            )
            
            # Add book info
            for seg in segments:
                seg["book"] = book_name
                seg["chapter"] = chapter_name
            
            book_segments.extend(segments)
            all_segments.extend(segments)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    book_stats[book_name] = len(book_segments)
    print(f"\n✅ {book_name} total: {len(book_segments)} segments")
    
    # Save individual book aligned data
    book_file = base_dir / f"{file_prefix}_aligned_whisper.json"
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

# Save combined aligned data
combined_file = base_dir / "all_books_aligned_whisper.json"
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
train_file = base_dir / "train_data_whisper.json"
val_file = base_dir / "val_data_whisper.json"

with open(train_file, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(val_file, 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print(f"✅ Train: {len(train_data)} segments → {train_file}")
print(f"✅ Val:   {len(val_data)} segments → {val_file}")

print("\n🎉 Done! Ready to train with Whisper-detected speech boundaries!")
print(f"\nNext: Update your training script to use:")
print(f"  - {train_file}")
print(f"  - {val_file}")