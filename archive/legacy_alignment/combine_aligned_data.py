#!/usr/bin/env python3

import json
import random
from pathlib import Path

print("="*60)
print("Combining Pre-Aligned Training Data")
print("="*60)

# ----------------------------------------------------
# Load All Pre-Aligned Book Data
# ----------------------------------------------------
print("\n📂 Loading pre-aligned data from individual books...")

aligned_files = [
    "matt_aligned_data.json",
    "mark_aligned_data.json",
    "james_aligned_data.json",
    # "acts_aligned_data.json",  # Uncomment when you have Acts
]

all_segments = []
book_stats = {}

for filename in aligned_files:
    file_path = Path(filename)
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Track which book this came from
        book_name = data[0].get("book", "Unknown") if data else "Unknown"
        book_stats[book_name] = len(data)
        
        all_segments.extend(data)
        print(f"✓ Loaded {len(data):4d} segments from {filename} ({book_name})")
    else:
        print(f"⚠️  {filename} not found - skipping")

if not all_segments:
    print("\n❌ ERROR: No aligned data found!")
    print("Make sure you've run the alignment script first.")
    exit(1)

print("\n" + "="*60)
print("📊 Dataset Summary")
print("="*60)
print(f"Total segments: {len(all_segments)}")
print("\nSegments per book:")
for book, count in sorted(book_stats.items()):
    percentage = (count / len(all_segments)) * 100
    print(f"  {book:12s}: {count:4d} segments ({percentage:5.1f}%)")

# ----------------------------------------------------
# Save Combined Dataset
# ----------------------------------------------------
print("\n💾 Saving combined dataset...")
with open("all_books_aligned_data.json", "w", encoding="utf-8") as f:
    json.dump(all_segments, f, ensure_ascii=False, indent=2)
print("✅ Saved all segments → all_books_aligned_data.json")

# ----------------------------------------------------
# Train / Val Split (80/20)
# ----------------------------------------------------
print("\n✂️  Splitting into train/validation sets...")
random.seed(42)
shuffled = all_segments.copy()
random.shuffle(shuffled)

split_point = int(len(shuffled) * 0.8)
train_data = shuffled[:split_point]
val_data = shuffled[split_point:]

# Save splits
with open("train_data.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("val_data.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print(f"✅ Train: {len(train_data):4d} segments ({len(train_data)/len(all_segments)*100:.1f}%) → train_data.json")
print(f"✅ Val:   {len(val_data):4d} segments ({len(val_data)/len(all_segments)*100:.1f}%) → val_data.json")

# Show distribution in training set
print("\n📊 Book distribution in training set:")
train_book_counts = {}
for item in train_data:
    book = item.get("book", "Unknown")
    train_book_counts[book] = train_book_counts.get(book, 0) + 1

for book in sorted(train_book_counts.keys()):
    count = train_book_counts[book]
    percentage = (count / len(train_data)) * 100
    print(f"  {book:12s}: {count:4d} segments ({percentage:5.1f}%)")

print("\n" + "="*60)
print("🎉 Done! Ready to train.")
print("="*60)
print("\nNext steps:")
print("  1. Use train_data.json for training")
print("  2. Use val_data.json for validation")
print("  3. Run your training script!")