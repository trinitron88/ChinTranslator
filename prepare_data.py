#!/usr/bin/env python3
"""
prepare_data.py — build the Hakha Chin (cnh) training set (V6).

Common Voice is PRE-ALIGNED: every clip is one short utterance with a
community-validated transcript. So there is no timestamp detection, no audio
slicing, no start/end bookkeeping — we just pair (audio array, sentence) and
save a Hugging Face DatasetDict that train.py loads directly.

`datasets>=4` dropped Common Voice's loader scripts, so we pull the raw files
(tsv transcripts + tar'd mp3 clips) from the public fsicoli mirror and decode
them ourselves into 16 kHz arrays stored in the Arrow dataset (self-contained,
no audio-backend needed at train time).

V6 data layout (vs V5, which pooled train+dev and re-split 80/20):

    train = official train + dev (ALL of it — no 20% thrown away)
            [+ vote-filtered `other` clips]  [+ pseudo-label-mined `other`]
            [+ Bible segments from the V4 archive, optional]
    val   = small carve-out from the train pool (early stopping / checkpoint
            selection during training)
    test  = the OFFICIAL Common Voice test split, untouched. 158 distinct
            speakers vs train's 6 — this is the honest "does it generalize to
            voices it never heard" number. Never train on it; evaluate_model.py
            reports WER/CER against it.

The `other` pool is Common Voice clips that never got enough review votes
(3,292 clips / 3.2 h — more audio than train+dev+test combined, 209 speakers).
Not validated ≠ wrong; most are fine. Two ways in:
  --include-other votes   clips with up_votes >= 1 and up >= down (precise)
  --other-keep FILE       keep-list from mine_other.py (pseudo-label filter:
                          a trained model agrees with the claimed transcript)

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
import json  # noqa: E402
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

SPLIT_TARS = {
    "train": f"audio/{LANG}/train/{LANG}_train_0.tar",
    "dev": f"audio/{LANG}/dev/{LANG}_dev_0.tar",
    "test": f"audio/{LANG}/test/{LANG}_test_0.tar",
    "other": f"audio/{LANG}/other/{LANG}_other_0.tar",
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


def read_tsv(split):
    tsv = fetch(f"transcript/{LANG}/{split}.tsv")
    with open(tsv, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def decode_rows(rows, clips, tag, limit=None):
    """Return [(16k float array, sentence), ...] for the given tsv rows."""
    n = len(rows) if limit is None else min(limit, len(rows))
    print(f"🔄 {tag}: decoding up to {n} clips ...")
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
    print(f"✓ {tag}: {len(records)} usable clips")
    return records


def _votes(row, key):
    try:
        return int(row.get(key) or 0)
    except ValueError:
        return 0


def select_other(rows, policy, keep_list):
    """Pick trustworthy rows from the unvalidated `other` pool.

    `other` clips have 0–1 review votes (never enough to validate). Vote
    filtering is high-precision / low-recall; a keep-list from mine_other.py
    (model-transcript agreement) recovers most of the rest.
    """
    picked, seen = [], set()

    def add(row):
        clip = Path(row.get("path", "")).name
        if clip and clip not in seen:
            seen.add(clip)
            picked.append(row)

    if policy == "votes":
        for r in rows:
            if _votes(r, "up_votes") >= 1 and _votes(r, "up_votes") >= _votes(r, "down_votes"):
                add(r)
    elif policy == "all":
        for r in rows:
            if _votes(r, "down_votes") <= _votes(r, "up_votes"):
                add(r)

    if keep_list:
        names = set(json.load(open(keep_list, encoding="utf-8")))
        for r in rows:
            if Path(r.get("path", "")).name in names:
                add(r)
        print(f"  other: +keep-list matched {len(picked)} total")

    return picked


def load_bible(json_path, audio_dir, limit=None):
    """Slice the archived V4 Bible alignment into (audio, sentence) records.

    90 min of clean male read speech in-language. It won't teach conversational
    style, but it strengthens the acoustic model of cnh phonology/orthography.
    """
    items = json.load(open(json_path, encoding="utf-8"))
    if limit:
        items = items[:limit]
    print(f"🔄 bible: slicing {len(items)} segments from {audio_dir} ...")
    records, missing = [], set()
    for it in items:
        path = Path(audio_dir) / Path(it["audio"]).name
        if not path.exists():
            missing.add(path.name)
            continue
        try:
            wav, _ = librosa.load(path, sr=TARGET_SR, mono=True,
                                  offset=it["start"], duration=it["end"] - it["start"])
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️  skip {path.name} [{it['start']:.1f}s]: {e}")
            continue
        if len(wav) / TARGET_SR < 0.3 or not it.get("text", "").strip():
            continue
        records.append((wav, it["text"].strip()))
    if missing:
        print(f"  ⚠️  {len(missing)} audio files not found (e.g. {sorted(missing)[:3]})")
    print(f"✓ bible: {len(records)} usable segments")
    return records


def main():
    ap = argparse.ArgumentParser(description="Build cnh DatasetDict for V6")
    ap.add_argument("--limit", type=int, help="max clips per source split (smoke test)")
    ap.add_argument("--out", default=OUT_DIR, help="save_to_disk path")
    ap.add_argument("--val-frac", type=float, default=0.08,
                    help="carve-out from the train pool for training-time eval "
                         "(the official test split stays untouched)")
    ap.add_argument("--include-other", choices=["off", "votes", "all"], default="votes",
                    help="add unvalidated `other` clips: votes = up>=1 and up>=down "
                         "(default), all = anything not downvoted, off = none")
    ap.add_argument("--other-keep",
                    help="JSON keep-list of `other` clip filenames from mine_other.py "
                         "(union'd with --include-other)")
    ap.add_argument("--bible-json",
                    help="archived V4 alignment JSON (e.g. archive/bible_dataset/"
                         "aligned_training_data.json) to mix in")
    ap.add_argument("--bible-audio-dir",
                    help="directory containing the Bible mp3s the JSON references "
                         "(e.g. /content/drive/MyDrive/ChinTranslator/Audio)")
    args = ap.parse_args()

    print("=" * 60)
    print(f"prepare_data.py — Common Voice {LANG}  (source: {REPO})")
    print("=" * 60)

    # ---- official validated splits ----
    train_pool = []
    for split in ("train", "dev"):
        rows = read_tsv(split)
        clips = extract_clips(fetch(SPLIT_TARS[split]), split)
        train_pool += decode_rows(rows, clips, split, limit=args.limit)

    test_rows = read_tsv("test")
    test_clips = extract_clips(fetch(SPLIT_TARS["test"]), "test")
    test_records = decode_rows(test_rows, test_clips, "test (held out)", limit=args.limit)

    # ---- unvalidated `other` pool ----
    if args.include_other != "off" or args.other_keep:
        rows = read_tsv("other")
        picked = select_other(rows, args.include_other, args.other_keep)
        print(f"  other: {len(picked)}/{len(rows)} rows selected "
              f"(policy={args.include_other}, keep-list={bool(args.other_keep)})")
        if picked:
            clips = extract_clips(fetch(SPLIT_TARS["other"]), "other")
            train_pool += decode_rows(picked, clips, "other", limit=args.limit)

    # ---- optional Bible mix-in ----
    if args.bible_json and args.bible_audio_dir:
        train_pool += load_bible(args.bible_json, args.bible_audio_dir, limit=args.limit)
    elif args.bible_json or args.bible_audio_dir:
        sys.exit("❌ --bible-json and --bible-audio-dir must be given together")

    # ---- split: train / val (carved) / test (official) ----
    random.seed(42)
    random.shuffle(train_pool)
    n_val = max(1, int(len(train_pool) * args.val_frac)) if args.val_frac > 0 else 0
    parts = {
        "train": train_pool[n_val:],
        "val": train_pool[:n_val],
        "test": test_records,
    }

    ds = DatasetDict()
    for name, recs in parts.items():
        if not recs:
            continue
        ds[name] = Dataset.from_dict({
            "audio": [a for a, _ in recs],          # 16 kHz float32 arrays
            "sentence": [s for _, s in recs],
        }, features=FEATURES)

    print(f"\n💾 Saving DatasetDict → {args.out}")
    ds.save_to_disk(args.out)

    total_min = sum(len(a) for a, _ in train_pool) / TARGET_SR / 60
    print("\n📊 V6 dataset")
    print(f"  train: {len(parts['train'])}   val: {len(parts['val'])}   "
          f"test: {len(parts['test'])} (official CV test — do not train on it)")
    print(f"  train+val audio: {total_min:.1f} min")
    print(f"  sample: {ds['train'][0]['sentence'][:70]}...")
    print("\n✅ Done. Load with: datasets.load_from_disk(%r)" % args.out)


if __name__ == "__main__":
    main()
