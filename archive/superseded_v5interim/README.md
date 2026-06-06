# Superseded V5-interim Scripts

Archived 2026-06-06. These were the first pass at adopting Common Voice, but
they bent the new pre-aligned data back into the old Bible-era schema
(`{audio, start, end, text}` + per-segment wavs) so the legacy training script
could consume it. The clean rewrite dropped that vestigial machinery.

Replaced by:
- `load_common_voice.py`  →  `../../prepare_data.py`
  (downloads cnh, builds a HF DatasetDict with a plain `{audio, sentence}`
  schema — no start/end, no hand-written wavs)
- `fine-tuning-new.py`    →  `../../train.py`
  (whisper-large-v3-turbo + LoRA + 8-bit, instead of full fine-tuning
  whisper-small)

Kept only for reference / history.
