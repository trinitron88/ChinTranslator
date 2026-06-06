#!/usr/bin/env python3
"""
prepare_data.py — build the Hakha Chin (cnh) training set for V5.

Common Voice is PRE-ALIGNED: every clip is one short utterance with a
community-validated transcript. So there is no timestamp detection, no audio
slicing, no start/end bookkeeping — we just pair (audio array, sentence) and
save a Hugging Face DatasetDict that train.py loads directly.

`datasets>=4` dropped Common Voice's loader scripts, so we pull the raw files
(tsv transcripts + tar'd mp3 clips) from the public fsicoli mirror and decode
them ourselves into 16 kHz arrays stored in the Arrow dataset (self-contained,
no audio-backend needed at train time).

Run on Colab:
    !python prepare_data.py
    # writes data/cv_cnh/  (load with datasets.load_from_disk)
"""

import subprocess
import sys


def _pip():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "datasets>=4", "huggingface_hub", "librosa", "soundfile"],
        check=False,
    )


_pip()

import argparse  # noqa: E402
import csv  # noqa: E402
import random  # noqa: E402
import tarfile  # noqa: E402
from pathlib import Path  # noqa: E402

import librosa  # noqa: E402
from datasets import Dataset, DatasetDict, Features, Sequence, Value  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

# Store decoded 16kHz audio as a plain float32 column (not the HF Audio feature,
# which needs torchcodec to encode). Keeps the dataset self-contained and the
# audio backend out of train time. train.py reads batch["audio"] as the array.
FEATURES = Features({"audio": Sequence(Value("float32")), "sentence": Value("string")})

REPO = "fsicoli/common_voice_17_0"  # public, raw tsv+tar layout
LANG = "cnh"
TARGET_SR = 16000
OUT_DIR = "data/cv_cnh"
CLIP_CACHE = Path("data/_clips")

# Common Voice ships official splits; we pool train+dev and 80/20 re-split
# (cnh's official dev split is abnormally large, ~as big as train).
SOURCE_SPLITS = {
    "train": f"audio/{LANG}/train/{LANG}_train_0.tar",
    "dev": f"audio/{LANG}/dev/{LANG}_dev_0.tar",
}


def fetch(path):
    print(f"  ↓ {path}")
    return hf_hub_download(REPO, path, repo_type="dataset")


def extract_clips(tar_path, split):
    dest = CLIP_CACHE / split
    dest.mkdir(parents=True, exist_ok=True)
    index = {}
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if m.name.lower().endswith(".mp3")]
        tar.extractall(dest, members=members, filter="data")
        for m in members:
            index[Path(m.name).name] = dest / m.name
    return index


def load_split(split, tar_repo_path, limit=None):
    """Return [(16k float array, sentence), ...] for one Common Voice split."""
    tsv = fetch(f"transcript/{LANG}/{split}.tsv")
    tar = fetch(tar_repo_path)
    clips = extract_clips(tar, split)

    with open(tsv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    n = len(rows) if limit is None else min(limit, len(rows))
    print(f"🔄 {split}: decoding up to {n} clips ...")

    records = []
    for i, row in enumerate(rows[:n]):
        sentence = (row.get("sentence") or "").strip()
        clip = Path(row.get("path", "")).name
        if not sentence or clip not in clips:
            continue
        try:
            wav, _ = librosa.load(clips[clip], sr=TARGET_SR, mono=True)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️  skip {clip}: {e}")
            continue
        if len(wav) / TARGET_SR < 0.3:
            continue
        records.append((wav, sentence))
        if (i + 1) % 250 == 0:
            print(f"  ... {i + 1}/{n}")

    print(f"✓ {split}: {len(records)} usable clips")
    return records


def main():
    ap = argparse.ArgumentParser(description="Build cnh DatasetDict for V5")
    ap.add_argument("--limit", type=int, help="max clips per source split (smoke test)")
    ap.add_argument("--out", default=OUT_DIR, help="save_to_disk path")
    ap.add_argument("--val-frac", type=float, default=0.2, help="validation fraction")
    args = ap.parse_args()

    print("=" * 60)
    print(f"prepare_data.py — Common Voice {LANG}  (source: {REPO})")
    print("=" * 60)

    pooled = []
    for split, tar_path in SOURCE_SPLITS.items():
        pooled += load_split(split, tar_path, limit=args.limit)

    random.seed(42)
    random.shuffle(pooled)
    cut = int(len(pooled) * (1 - args.val_frac))
    parts = {"train": pooled[:cut], "test": pooled[cut:]}

    ds = DatasetDict()
    for name, recs in parts.items():
        ds[name] = Dataset.from_dict({
            "audio": [a for a, _ in recs],          # 16 kHz float32 arrays
            "sentence": [s for _, s in recs],
        }, features=FEATURES)

    print(f"\n💾 Saving DatasetDict → {args.out}")
    ds.save_to_disk(args.out)

    total_min = sum(len(a) for a, _ in pooled) / TARGET_SR / 60
    print("\n📊 V5 dataset")
    print(f"  train: {len(parts['train'])}   test: {len(parts['test'])}")
    print(f"  audio: {total_min:.1f} min")
    print(f"  sample: {ds['train'][0]['sentence'][:70]}...")
    print("\n✅ Done. Load with: datasets.load_from_disk(%r)" % args.out)


if __name__ == "__main__":
    main()
