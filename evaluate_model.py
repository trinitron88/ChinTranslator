#!/usr/bin/env python3
"""
evaluate_model.py — measure WER/CER of a served model on the held-out test set.

Runs the SAME engine the apps serve with (faster-whisper / CTranslate2), so the
number you get here is the number users experience. Evaluates against the
official Common Voice cnh test split that prepare_data.py stores untouched in
data/cv_cnh (763 clips, 158 speakers the model never heard).

Reference point: the best published Common Voice cnh result is ~31.4% WER
(gchhablani/wav2vec2-large-xlsr-cnh) on this same split.

Usage:
    python evaluate_model.py --model whisper-cnh-turbo-ct2
    python evaluate_model.py --model large-v3-turbo          # stock baseline
    python evaluate_model.py --model A --model B             # compare two
"""

import subprocess
import sys


def _pip():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "faster-whisper", "datasets>=4", "jiwer"],
        check=False,
    )


_pip()

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402


def normalize(text: str) -> str:
    """Case + punctuation normalization; keeps cnh diacritics (ṭ, etc.)."""
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def model_language(model_dir: str, override: str | None) -> str | None:
    """The surrogate language token the model was trained with (V6+). Stored in
    chin_metadata.json by train.py/export_model.py; None = V5 task-only mode."""
    if override:
        return None if override == "none" else override
    meta = Path(model_dir) / "chin_metadata.json"
    if meta.is_file():
        try:
            return json.loads(meta.read_text(encoding="utf-8")).get("language_token")
        except Exception:  # noqa: BLE001
            pass
    return None


def transcribe_dataset(model_path, rows, language, beam=5, log_every=50):
    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⏳ Loading {model_path} on {device} "
          f"(language={language or 'auto-detect'}) ...")
    model = WhisperModel(model_path, device=device,
                         compute_type="float16" if device == "cuda" else "int8")

    hyps, t0 = [], time.time()
    for i, ex in enumerate(rows):
        wav = np.asarray(ex["audio"], dtype=np.float32)
        kwargs = dict(task="transcribe", beam_size=beam,
                      condition_on_previous_text=False,
                      temperature=[0.0, 0.2, 0.4],
                      compression_ratio_threshold=2.2, vad_filter=False)
        if language:
            kwargs["language"] = language
        segs, _ = model.transcribe(wav, **kwargs)
        hyps.append(" ".join(s.text for s in segs).strip())
        if (i + 1) % log_every == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  ... {i + 1}/{len(rows)}  ({rate:.1f} clips/s)")
    return hyps


def score(refs, hyps):
    import jiwer
    n_refs = [normalize(r) for r in refs]
    n_hyps = [normalize(h) for h in hyps]
    return {
        "wer_raw": 100 * jiwer.wer(refs, hyps),
        "wer_norm": 100 * jiwer.wer(n_refs, n_hyps),
        "cer_norm": 100 * jiwer.cer(n_refs, n_hyps),
    }


def main():
    ap = argparse.ArgumentParser(description="WER/CER on the held-out CV cnh test split")
    ap.add_argument("--model", action="append", required=True,
                    help="CT2 dir / HF repo / built-in size; repeat to compare")
    ap.add_argument("--data", default="data/cv_cnh")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, help="evaluate only the first N clips")
    ap.add_argument("--language", help="force a language token ('none' = auto); "
                                       "default reads the model's chin_metadata.json")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--report", help="write per-utterance results to this JSON file")
    args = ap.parse_args()

    from datasets import load_from_disk
    ds = load_from_disk(args.data)
    if args.split not in ds:
        sys.exit(f"❌ split '{args.split}' not in {args.data} (have {list(ds)}). "
                 f"Re-run the V6 prepare_data.py to get a held-out test split.")
    rows = ds[args.split]
    if args.limit:
        rows = rows.select(range(min(args.limit, len(rows))))
    refs = [ex["sentence"] for ex in rows]
    print(f"📂 {args.data}:{args.split} — {len(rows)} clips")

    results = {}
    for model_path in args.model:
        lang = model_language(model_path, args.language)
        hyps = transcribe_dataset(model_path, rows, lang, beam=args.beam)
        metrics = score(refs, hyps)
        results[model_path] = (metrics, hyps)
        print(f"\n📈 {model_path}")
        print(f"   WER {metrics['wer_raw']:.1f}% raw | {metrics['wer_norm']:.1f}% "
              f"normalized | CER {metrics['cer_norm']:.1f}%")

        # worst offenders — the fastest way to see WHAT the model gets wrong
        import jiwer
        per_utt = sorted(
            ((jiwer.cer(normalize(r), normalize(h) or " "), r, h)
             for r, h in zip(refs, hyps)),
            reverse=True)
        print("   worst 5:")
        for cer, r, h in per_utt[:5]:
            print(f"     [{100 * cer:5.1f}%] ref: {r[:70]}")
            print(f"              hyp: {h[:70]}")

    if len(results) > 1:
        print("\n🏁 Comparison (normalized WER):")
        for m, (metrics, _) in sorted(results.items(), key=lambda kv: kv[1][0]["wer_norm"]):
            print(f"   {metrics['wer_norm']:6.1f}%  {m}")

    if args.report:
        payload = {
            m: {"metrics": metrics,
                "utterances": [{"ref": r, "hyp": h} for r, h in zip(refs, hyps)]}
            for m, (metrics, hyps) in results.items()
        }
        Path(args.report).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
        print(f"📝 report → {args.report}")


if __name__ == "__main__":
    main()
