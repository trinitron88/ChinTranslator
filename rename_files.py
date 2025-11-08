"""
Batch File Renamer for Bible Chapters
======================================
Renames files from long formats to short abbreviations.

Examples:
- matthew_01.txt → matt_01.txt
- matthew_chapter_1.mp3 → matt_01.mp3
- mark_chapter_01.txt → mark_01.txt
- luke_01.mp3 → luke_01.mp3

Usage:
1. Put this script in the same directory as your files
2. Configure the RENAME_RULES below
3. Run: python batch_rename.py
4. Confirm the preview before actually renaming
"""

import os
from pathlib import Path
import re

# ============ CONFIGURATION ============

# Define your rename rules
# Format: "long_name" -> "short_name"
RENAME_RULES = {
    "matthew": "matt",
    "mark": "mark",  # Already short, but included for completeness
    "luke": "luke",
    "john": "john",
    "james": "james",
    "genesis": "gen",
    "exodus": "exod",
    "leviticus": "lev",
    "numbers": "num",
    "deuteronomy": "deut",
    # Add more books as needed...
}

# Directories to process
AUDIO_DIR = Path("Audio")
TEXT_DIR = Path("Text")

# ========================================

def extract_book_and_number(filename):
    """
    Extract book name and chapter number from filename.
    Handles formats like:
    - matthew_01.txt
    - matthew_chapter_1.mp3
    - Matthew 1.txt
    - matthew-chapter-01.mp3
    """
    name_lower = filename.lower()
    
    # Try to find a book name
    book = None
    for long_name in RENAME_RULES.keys():
        if long_name in name_lower:
            book = long_name
            break
    
    if not book:
        return None, None
    
    # Try to extract chapter number
    # Look for patterns like: _01, _1, -01, -1, " 1", "chapter_1", etc.
    patterns = [
        r'[_\-\s](\d{1,3})',           # _01, -01, " 1"
        r'chapter[_\-\s]?(\d{1,3})',   # chapter_01, chapter-1
        r'ch[_\-\s]?(\d{1,3})',        # ch_01, ch-1
    ]
    
    chapter_num = None
    for pattern in patterns:
        match = re.search(pattern, name_lower)
        if match:
            chapter_num = int(match.group(1))
            break
    
    return book, chapter_num

def generate_new_name(old_name, book, chapter_num, extension):
    """Generate new standardized filename."""
    short_book = RENAME_RULES.get(book, book)
    new_name = f"{short_book}_{chapter_num:02d}{extension}"
    return new_name

def preview_renames(directory):
    """Show what would be renamed without actually doing it."""
    if not directory.exists():
        print(f"⚠️  Directory not found: {directory}")
        return []
    
    rename_list = []
    
    for filepath in sorted(directory.iterdir()):
        if filepath.is_file():
            old_name = filepath.name
            extension = filepath.suffix
            name_without_ext = filepath.stem
            
            book, chapter_num = extract_book_and_number(old_name)
            
            if book and chapter_num:
                new_name = generate_new_name(old_name, book, chapter_num, extension)
                
                if old_name != new_name:
                    rename_list.append({
                        'old_path': filepath,
                        'old_name': old_name,
                        'new_name': new_name,
                        'book': book,
                        'chapter': chapter_num
                    })
    
    return rename_list

def perform_renames(rename_list):
    """Actually rename the files."""
    success_count = 0
    error_count = 0
    
    for item in rename_list:
        old_path = item['old_path']
        new_path = old_path.parent / item['new_name']
        
        try:
            # Check if target already exists
            if new_path.exists():
                print(f"⚠️  SKIP: {item['new_name']} already exists")
                error_count += 1
                continue
            
            old_path.rename(new_path)
            print(f"✅ {item['old_name']} → {item['new_name']}")
            success_count += 1
        
        except Exception as e:
            print(f"❌ ERROR renaming {item['old_name']}: {e}")
            error_count += 1
    
    return success_count, error_count

# ============ MAIN SCRIPT ============

print("="*60)
print("📝 Bible Chapter File Renamer")
print("="*60)

# Preview all renames
print("\n🔍 Scanning directories...")
audio_renames = preview_renames(AUDIO_DIR)
text_renames = preview_renames(TEXT_DIR)

all_renames = audio_renames + text_renames

if not all_renames:
    print("\n✅ No files need renaming! Everything looks good.")
    exit(0)

# Show preview
print(f"\n📋 Found {len(all_renames)} files to rename:\n")

print("AUDIO FILES:")
for item in audio_renames:
    print(f"  {item['old_name']:40} → {item['new_name']}")

print("\nTEXT FILES:")
for item in text_renames:
    print(f"  {item['old_name']:40} → {item['new_name']}")

# Confirm
print("\n" + "="*60)
response = input("Proceed with renaming? (yes/no): ").strip().lower()

if response not in ['yes', 'y']:
    print("❌ Cancelled. No files were renamed.")
    exit(0)

# Perform renames
print("\n🔄 Renaming files...\n")
success, errors = perform_renames(all_renames)

print("\n" + "="*60)
print("✅ COMPLETE!")
print("="*60)
print(f"Success: {success} files")
if errors > 0:
    print(f"Errors: {errors} files")
print("\n💡 Tip: Run this script again before each training run to keep filenames consistent!")