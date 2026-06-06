#!/usr/bin/env python3
"""
Download Mozilla Common Voice (Hakha Chin / cnh) and convert it into the same
segment schema the rest of the pipeline uses:

    {"audio": <path>, "start": 0.0, "end": <duration>, "text": <sentence>,
     "book": "CommonVoice", "gender": ..., "age": ...}

Each Common Voice clip is already a single short utterance with a clean,
community-validated transcript, so there is NO alignment step -- every clip
becomes one segment with start=0.0 and end=duration.

This adds ~300 speakers of mixed gender/age and everyday (non-biblical)
vocabulary, directly addressing the V4 limitations (male-only Bible narration,
religious register only). It is still *scripted* speech, not free conversation.

`datasets>=4` dropped support for Common Voice's loader scripts, so we pull the
raw files (tsv transcripts + tar'd mp3 clips) straight from the Hugging Face
mirror and parse them ourselves. The fsicoli mirror is public (ungated).

Usage:
    python load_common_voice.py                 # download + convert (defaults)
    python load_common_voice.py --limit 20      # quick smoke test
    python load_common_voice.py --merge aligned_train_data.json aligned_val_data.json
"""

import argparse
import csv
import json
import random
import tarfile
from pathlib import Path

import librosa
import soundfile as sf
from huggingface_hub import hf_hub_download

REPO = "fsicoli/common_voice_17_0"  # raw tsv+tar layout, ungated
LANG = "cnh"
TARGET_SR = 16000
AUDIO_ROOT = Path("Audio/common_voice")

# Common Voice split (tsv basename, tar path) -> our role.
# train.tsv = official train split; dev.tsv = official validation split.
SPLITS = {
    "train": ("train", f"audio/{LANG}/train/{LANG}_train_0.tar"),
    "val": ("dev", f"audio/{LANG}/dev/{LANG}_dev_0.tar"),
}


def fetch(path):
    print(f"  ↓ {path}")
    return hf_hub_download(REPO, path, repo_type="dataset")


def extract_clips(tar_path, role):
    """Extract a split's mp3 clips, return {basename: extracted_path}."""
    dest = AUDIO_ROOT / "_clips" / role
    dest.mkdir(parents=True, exist_ok=True)
    index = {}
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if m.name.lower().endswith(".mp3")]
        tar.extractall(dest, members=members, filter="data")
        for m in members:
            index[Path(m.name).name] = dest / m.name
    print(f"  ✓ extracted {len(index)} clips for '{role}'")
    return index


def convert_split(role, tsv_name, tar_repo_path, limit=None):
    """Pair tsv rows with extracted clips, write 16kHz wavs, return segments."""
    tsv_path = fetch(f"transcript/{LANG}/{tsv_name}.tsv")
    tar_path = fetch(tar_repo_path)
    clip_index = extract_clips(tar_path, role)

    out_dir = AUDIO_ROOT / role
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = []
    with open(tsv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    n = len(rows) if limit is None else min(limit, len(rows))
    print(f"\n🔄 Converting up to {n} clips for '{role}' -> {out_dir}/ ...")

    for i, row in enumerate(rows[:n]):
        sentence = (row.get("sentence") or "").strip()
        clip_name = Path(row.get("path", "")).name
        if not sentence or clip_name not in clip_index:
            continue

        try:
            wav, _ = librosa.load(clip_index[clip_name], sr=TARGET_SR, mono=True)
        except Exception as e:  # noqa: BLE001 - skip the odd unreadable clip
            print(f"  ⚠️  skip {clip_name}: {e}")
            continue

        duration = len(wav) / TARGET_SR
        if duration < 0.3:
            continue

        out_path = out_dir / f"cnh_{role}_{i:05d}.wav"
        sf.write(out_path, wav, TARGET_SR)

        segments.append({
            "audio": str(out_path),
            "start": 0.0,
            "end": round(duration, 3),
            "text": sentence,
            "book": "CommonVoice",
            "gender": row.get("gender") or "unknown",
            "age": row.get("age") or "unknown",
        })

        if (i + 1) % 250 == 0:
            print(f"  ... {i + 1}/{n}")

    print(f"✓ {len(segments)} segments written for '{role}'")
    return segments


def diversity_report(segments):
    genders, ages = {}, {}
    for s in segments:
        genders[s["gender"]] = genders.get(s["gender"], 0) + 1
        ages[s["age"]] = ages.get(s["age"], 0) + 1
    total_sec = sum(s["end"] - s["start"] for s in segments)
    print("\n📊 Common Voice diversity:")
    print(f"  Total clips:    {len(segments)}")
    print(f"  Total audio:    {total_sec / 60:.1f} min")
    print("  Gender: " + ", ".join(f"{k}={v}" for k, v in sorted(genders.items())))
    print("  Age:    " + ", ".join(f"{k}={v}" for k, v in sorted(ages.items())))


def resplit_cv_only(cv_train, cv_val, ratio=0.8):
    """Pool all Common Voice clips and re-split, since cnh's official dev split
    is nearly as large as train (wasteful). V5 = Common Voice only."""
    pooled = cv_train + cv_val
    random.seed(42)
    random.shuffle(pooled)
    cut = int(len(pooled) * ratio)
    train, val = pooled[:cut], pooled[cut:]

    with open("v5_train_data.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open("v5_val_data.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print(f"\n🧬 V5 dataset (Common Voice only, {int(ratio*100)}/{int((1-ratio)*100)} split):")
    print(f"  Train: {len(train)} → v5_train_data.json")
    print(f"  Val:   {len(val)} → v5_val_data.json")


def merge_v5(cv_train, cv_val, bible_train_path, bible_val_path):
    """Combine Common Voice segments with existing Bible segments into V5 files."""
    with open(bible_train_path, encoding="utf-8") as f:
        bible_train = json.load(f)
    with open(bible_val_path, encoding="utf-8") as f:
        bible_val = json.load(f)

    train = bible_train + cv_train
    val = bible_val + cv_val
    random.seed(42)
    random.shuffle(train)
    random.shuffle(val)

    with open("v5_train_data.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open("v5_val_data.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print("\n🧬 V5 merged dataset:")
    print(f"  Train: {len(bible_train)} Bible + {len(cv_train)} CV = {len(train)} → v5_train_data.json")
    print(f"  Val:   {len(bible_val)} Bible + {len(cv_val)} CV = {len(val)} → v5_val_data.json")


def main():
    ap = argparse.ArgumentParser(description="Download + convert Common Voice cnh")
    ap.add_argument("--limit", type=int, help="max clips per split (smoke test)")
    ap.add_argument("--merge", nargs=2, metavar=("BIBLE_TRAIN", "BIBLE_VAL"),
                    help="also merge with existing Bible aligned files into V5")
    ap.add_argument("--resplit", type=float, nargs="?", const=0.8, metavar="RATIO",
                    help="V5 = Common Voice only; pool train+dev and re-split (default 0.8)")
    args = ap.parse_args()

    print("=" * 60)
    print(f"Common Voice (Hakha Chin / {LANG}) -> aligned segment schema")
    print(f"Source: {REPO}")
    print("=" * 60)

    cv = {}
    for role, (tsv_name, tar_path) in SPLITS.items():
        cv[role] = convert_split(role, tsv_name, tar_path, limit=args.limit)

    cv_train = cv.get("train", [])
    cv_val = cv.get("val", [])

    with open("commonvoice_train_data.json", "w", encoding="utf-8") as f:
        json.dump(cv_train, f, ensure_ascii=False, indent=2)
    with open("commonvoice_val_data.json", "w", encoding="utf-8") as f:
        json.dump(cv_val, f, ensure_ascii=False, indent=2)
    print("\n💾 Saved commonvoice_train_data.json / commonvoice_val_data.json")

    diversity_report(cv_train + cv_val)

    if args.resplit is not None:
        resplit_cv_only(cv_train, cv_val, ratio=args.resplit)
    elif args.merge:
        merge_v5(cv_train, cv_val, args.merge[0], args.merge[1])

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
