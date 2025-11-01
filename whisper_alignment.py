"""
Use Whisper to get timestamps and align with Hakha Chin text
This creates properly chunked training data for ALL chapters (Mark + Matthew)
"""
import whisper
import json
from pathlib import Path

print("="*60)
print("Creating Aligned Training Data - ALL CHAPTERS")
print("Mark (16 chapters) + Matthew (28 chapters) = 44 total")
print("="*60)

# Load pre-trained Whisper model
print("\n📥 Loading Whisper model...")
model = whisper.load_model("tiny")  # Using 'tiny' for speed - can use 'base' or 'small' for better accuracy
print("✅ Model loaded")

# Load your original paired data
print("\n📂 Loading paired data...")

# Try to load both Mark and Matthew data
all_chapters = []

# Load Mark chapters if they exist
mark_file = Path('training_data.json')
if mark_file.exists():
    with open(mark_file, 'r', encoding='utf-8') as f:
        mark_data = json.load(f)
    all_chapters.extend(mark_data)
    print(f"✓ Loaded {len(mark_data)} Mark chapters")
else:
    print("⚠️  No Mark training_data.json found")

# Load Matthew chapters if they exist
matt_file = Path('matthew_training_data.json')
if matt_file.exists():
    with open(matt_file, 'r', encoding='utf-8') as f:
        matt_data = json.load(f)
    all_chapters.extend(matt_data)
    print(f"✓ Loaded {len(matt_data)} Matthew chapters")
else:
    print("⚠️  No Matthew matthew_training_data.json found")

if not all_chapters:
    print("❌ ERROR: No chapter data found!")
    print("Make sure training_data.json or matthew_training_data.json exists")
    exit(1)

print(f"\n📊 Total chapters to process: {len(all_chapters)}")

# Process each chapter
aligned_segments = []
failed_chapters = []

for idx, chapter in enumerate(all_chapters, 1):
    chapter_name = chapter['chapter']
    print(f"\n[{idx}/{len(all_chapters)}] 🔄 Processing {chapter_name}...")
    
    audio_path = chapter['audio']
    full_text = chapter['text']
    
    try:
        # Transcribe with timestamps using Whisper
        result = model.transcribe(
            audio_path,
            language=None,  # Let Whisper detect the language
            task="transcribe",
            verbose=False
        )
        
        print(f"    Found {len(result['segments'])} Whisper segments")
        
        # Split text into sentences (rough split by periods)
        sentences = [s.strip() for s in full_text.split('.') if s.strip()]
        print(f"    Split into {len(sentences)} sentences")
        
        # Match Whisper segments with sentences
        # Simple approach: distribute sentences across segments
        if len(sentences) > 0 and len(result['segments']) > 0:
            sentences_per_segment = max(1, len(sentences) // len(result['segments']))
            
            sentence_idx = 0
            segments_created = 0
            
            for seg in result['segments']:
                # Take a few sentences for this segment
                end_idx = min(sentence_idx + sentences_per_segment, len(sentences))
                segment_text = '. '.join(sentences[sentence_idx:end_idx])
                
                if segment_text:  # Only add if we have text
                    aligned_segments.append({
                        'audio': audio_path,
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': segment_text,
                        'chapter': chapter_name
                    })
                    segments_created += 1
                
                sentence_idx = end_idx
                
                if sentence_idx >= len(sentences):
                    break
            
            print(f"    ✅ Created {segments_created} aligned segments")
        else:
            print(f"    ⚠️  Warning: No sentences or segments found")
            failed_chapters.append(chapter_name)
    
    except Exception as e:
        print(f"    ❌ Error processing {chapter_name}: {e}")
        failed_chapters.append(chapter_name)

print("\n" + "="*60)
print("PROCESSING COMPLETE")
print("="*60)
print(f"✅ Total aligned segments: {len(aligned_segments)}")
print(f"✅ Successfully processed: {len(all_chapters) - len(failed_chapters)} chapters")

if failed_chapters:
    print(f"⚠️  Failed chapters: {len(failed_chapters)}")
    for chapter in failed_chapters:
        print(f"   - {chapter}")

# Show sample
print("\n📋 Sample aligned segment:")
if aligned_segments:
    sample = aligned_segments[0]
    print(f"  Chapter: {sample['chapter']}")
    print(f"  Time: {sample['start']:.2f}s - {sample['end']:.2f}s")
    print(f"  Duration: {sample['end'] - sample['start']:.2f}s")
    print(f"  Text: {sample['text'][:100]}...")

# Show statistics
if aligned_segments:
    durations = [seg['end'] - seg['start'] for seg in aligned_segments]
    avg_duration = sum(durations) / len(durations)
    print(f"\n📊 Segment statistics:")
    print(f"  Average duration: {avg_duration:.2f}s")
    print(f"  Min duration: {min(durations):.2f}s")
    print(f"  Max duration: {max(durations):.2f}s")

# Save aligned data
output_file = 'all_chapters_aligned_data.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(aligned_segments, f, ensure_ascii=False, indent=2)

print(f"\n💾 Saved to: {output_file}")

# Now split into train/val
print("\n" + "="*60)
print("Splitting into train/validation sets...")
print("="*60)

import random
random.seed(42)

# Shuffle the segments
shuffled = aligned_segments.copy()
random.shuffle(shuffled)

# 80/20 split
split_point = int(len(shuffled) * 0.8)
train_data = shuffled[:split_point]
val_data = shuffled[split_point:]

# Save splits
train_file = 'aligned_train_data.json'
val_file = 'aligned_val_data.json'

with open(train_file, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(val_file, 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print(f"✅ Training set: {len(train_data)} segments → {train_file}")
print(f"✅ Validation set: {len(val_data)} segments → {val_file}")

print("\n" + "="*60)
print("🎉 ALL DONE!")
print("="*60)
print("\nNext steps:")
print("1. Review the aligned data files to verify quality")
print("2. Run your fine-tuning script using:")
print(f"   - Training: {train_file}")
print(f"   - Validation: {val_file}")
print("3. Train your improved model!")
print("\n✨ Ready to create whisper-hakha-chin-v4!")