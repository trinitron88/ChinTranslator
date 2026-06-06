# Legacy Alignment / Bible-prep Scripts (pre-V5)

Archived 2026-06-06. These scripts built training data from raw Bible audio by
detecting timestamps and slicing audio into aligned segments. **V5 doesn't need
any of this** — it trains on Mozilla Common Voice, where every clip is already a
single utterance with a validated transcript (see `../../load_common_voice.py`).

Kept (not deleted) because if you ever collect raw *conversational* Hakha Chin
audio, that audio WILL need alignment again, and these are a working starting point.

## Contents
- `whisper_alignment_2.py`, `whisper_alignment_w_timestamps.py` — Whisper-based timestamp alignment
- `dumb_split.py` — naive fixed-duration splitter (caused the V3 multilingual-gibberish disaster; kept as a cautionary example, do NOT reuse for training)
- `process-input.py` — multi-book pair-audio-with-text + align workflow
- `combine_aligned_data.py` — merged per-book aligned JSONs + train/val split (replaced by load_common_voice.py's --resplit/--merge)
- `rename_files.py` — batch renamer for Bible chapter files
- `segment_checker.py` — QA: play back aligned segments in a notebook
- `vad.py` — silero-VAD segmentation + faster-whisper transcription (standalone; was NOT used by gradio_interface.py, which inlines its own VAD)
- `fine-tuning.py` — original training script, superseded by ../../fine-tuning-aligned.py
