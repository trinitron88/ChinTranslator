#!/usr/bin/env python3
"""
mine_other.py — rescue training data from Common Voice's unvalidated pool.

The cnh `other` split is 3,292 clips / 3.2 hours / 209 speakers — MORE audio
than train+dev+test combined — that simply never got enough review votes.
Most of it is fine; some is mispronounced, truncated, or noise.

This script pseudo-label-filters it: transcribe every `other` clip with your
current fine-tuned model and keep the ones where the model's transcript agrees
with the claimed one (low character error rate). If an independent model reads
the audio and produces the labeled sentence, the label is almost certainly
right — and a clip the model gets *slightly* wrong is exactly the training
signal we want, so the default threshold is forgiving (CER ≤ 15%).

This is the data flywheel:
    train V6 → mine_other.py → prepare_data.py --other-keep keep.json →
    retrain (V6.1, more data + more voices) → mine again with the better model

Usage (Colab, after export_model.py):
    !python mine_other.py --model /path/to/whisper-cnh-turbo-ct2
    !python prepare_data.py --other-keep other_keep.json
"""

import subprocess
import sys


def _pip():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "faster-whisper", "huggingface_hub", "librosa", "jiwer"],
        check=False,
    )


_pip()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

# reuse the Common Voice download/extract helpers (and their fsicoli paths)
from prepare_data import SPLIT_TARS, TARGET_SR, extract_clips, fetch, read_tsv  # noqa: E402
from evaluate_model import model_language, normalize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Pseudo-label filter the CV cnh `other` pool")
    ap.add_argument("--model", required=True, help="CT2 model dir (from export_model.py)")
    ap.add_argument("--max-cer", type=float, default=0.15,
                    help="keep clips whose model-vs-label CER is at or below this")
    ap.add_argument("--out", default="other_keep.json", help="keep-list for prepare_data.py")
    ap.add_argument("--review", default="other_mined.tsv",
                    help="full per-clip results (clip, cer, label, hypothesis) for eyeballing")
    ap.add_argument("--language", help="override the model's stored language token")
    ap.add_argument("--limit", type=int, help="only process the first N clips (smoke test)")
    args = ap.parse_args()

    import librosa
    import jiwer
    import torch
    from faster_whisper import WhisperModel

    rows = read_tsv("other")
    if args.limit:
        rows = rows[:args.limit]
    clips = extract_clips(fetch(SPLIT_TARS["other"]), "other")

    lang = model_language(args.model, args.language)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⏳ Loading {args.model} on {device} (language={lang or 'auto'}) ...")
    model = WhisperModel(args.model, device=device,
                         compute_type="float16" if device == "cuda" else "int8")

    keep, results, t0 = [], [], time.time()
    for i, row in enumerate(rows):
        sentence = (row.get("sentence") or "").strip()
        clip = Path(row.get("path", "")).name
        if not sentence or clip not in clips:
            continue
        try:
            wav, _ = librosa.load(clips[clip], sr=TARGET_SR, mono=True)
        except Exception:  # noqa: BLE001
            continue
        if len(wav) / TARGET_SR < 0.3:
            continue
        kwargs = dict(task="transcribe", beam_size=5,
                      condition_on_previous_text=False, vad_filter=False)
        if lang:
            kwargs["language"] = lang
        segs, _ = model.transcribe(wav, **kwargs)
        hyp = " ".join(s.text for s in segs).strip()
        ref_n, hyp_n = normalize(sentence), normalize(hyp)
        cer = jiwer.cer(ref_n, hyp_n or " ") if ref_n else 1.0
        results.append((clip, cer, sentence, hyp))
        if cer <= args.max_cer:
            keep.append(clip)
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  ... {i + 1}/{len(rows)}  kept {len(keep)}  ({rate:.1f} clips/s)")

    Path(args.out).write_text(json.dumps(keep, indent=1), encoding="utf-8")
    with open(args.review, "w", encoding="utf-8") as f:
        f.write("clip\tcer\tlabel\thypothesis\n")
        for clip, cer, ref, hyp in sorted(results, key=lambda r: r[1]):
            f.write(f"{clip}\t{cer:.3f}\t{ref}\t{hyp}\n")

    print(f"\n✅ kept {len(keep)}/{len(results)} clips (CER ≤ {args.max_cer:.0%})")
    print(f"   keep-list → {args.out}")
    print(f"   full results → {args.review} (sorted by CER; skim the boundary cases)")
    print(f"\nNext: python prepare_data.py --other-keep {args.out}")


if __name__ == "__main__":
    main()
