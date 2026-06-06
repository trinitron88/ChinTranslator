# Archived Bible Dataset (pre-V5)

The original Hakha Chin training data used for models V1–V4, archived 2026-06-06
when V5 switched to **Mozilla Common Voice only** (see `load_common_voice.py`
and the project handoff).

## Contents
- `aligned_train_data.json` / `aligned_val_data.json` — Mark-only aligned segments (540 / 136)
- `aligned_training_data.json` — combined Mark set (676)
- `audio/` — raw Bible mp3s: Mark (16), Matthew (28), James (5) — Faith Comes By Hearing
- `text/` — matching transcripts (YouVersion), UTF-8

Note: the aligned JSONs reference paths like `Audio/mark_05.mp3`. To re-use this
data, move the mp3s back to `Audio/` (or fix the paths) before training/merging.

## To bring it back into training
```bash
# move audio back so the paths resolve, then merge with Common Voice:
mv archive/bible_dataset/audio/*.mp3 Audio/
python load_common_voice.py --merge archive/bible_dataset/aligned_train_data.json \
                                    archive/bible_dataset/aligned_val_data.json
```
