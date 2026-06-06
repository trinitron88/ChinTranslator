# Session Memory — 2026-06-06 (Hakha Chin STT, V5)

A working log of what we changed, why, and how to use it next time. Branch:
`claude/latest-result-5kTMp`.

## TL;DR
- Fixed the `Invalid model size 'whisper-cnh-turbo-ct2'` crash in the Gradio app.
- Found the real reason "the latest model wasn't in Drive": the V5 scripts
  saved to **ephemeral `/content`**, which Colab wipes on runtime reset.
- Made training/export **persist to Drive by default** on Colab.
- Added a **one-run Colab notebook** for a clean fresh cycle, plus an
  **emergency rescue script** that builds a live demo from a saved checkpoint
  without retraining.

## The original bug
The Gradio app errored with:
> Invalid model size 'whisper-cnh-turbo-ct2', expected one of: tiny, base, ...

**Cause:** faster-whisper only accepts a built-in size name, an HF repo id
(`org/model`), or an *existing* local directory. `CHIN_MODEL=whisper-cnh-turbo-ct2`
is a bare local folder name; when that folder isn't found in the working dir,
faster-whisper falls back to treating it as a size name and rejects it.

**Fix (`gradio_interface.py`):**
- Added `resolve_model()`: existing dir → absolute path; built-in size → pass
  through; HF repo id → pass through; otherwise **fail fast** with an actionable
  message at startup.
- Load the model **once at startup** instead of per request (fails fast, faster
  transcription).

## The "where did my model go?" issue
The V5 pipeline is fully self-contained and needs **no Drive data to train** —
`prepare_data.py` pulls Common Voice `cnh` from the public HF mirror
`fsicoli/common_voice_17_0`. The only problem was **persistence**:

- `train.py` → `whisper-cnh-turbo-lora` (relative)
- `export_model.py` → `whisper-cnh-turbo-ct2` (relative)

On Colab the working dir is `/content` (ephemeral, **not** Drive), so outputs
vanished on runtime reset and never appeared in Drive. (Only the older
`ChinTranslator_Colab.ipynb` wrote to Drive.)

**Fix:** both `train.py` and `export_model.py` now default their output dirs to
`/content/drive/MyDrive/ChinTranslator` when Drive is mounted, else the current
dir. Explicit CLI args still win.

## New files added this session
- **`ChinTranslator_V5_Colab.ipynb`** — phone-friendly run-all: GPU check →
  mount Drive → clone repo → prepare data → train → export to CT2 → launch
  Gradio. Saves everything to a versioned `runs/<timestamp>/` folder and writes
  `LATEST_MODEL.txt`. Has a `SMOKE` toggle for a fast dry run.
- **`ChinTranslator_V5_Colab.txt`** / **`..._ONECELL.txt`** — plain-text exports
  (the one-cell version pastes into a single Colab cell).
- **`rescue_export.py`** — emergency: finds the newest LoRA epoch-checkpoint
  across all Drive runs, merges → CT2 → saves to Drive → launches Gradio. No
  retraining. One-line invocation:
  ```
  !rm -rf /content/CT && git clone -q -b claude/latest-result-5kTMp \
    https://github.com/trinitron88/ChinTranslator.git /content/CT && \
    python /content/CT/rescue_export.py
  ```

## Key learnings / gotchas
- **Colab `/content` is ephemeral.** Anything you want to keep must be written
  under `/content/drive/MyDrive/...` (and Drive must be mounted).
- **`save_strategy="epoch"`** in `train.py` means a checkpoint lands in the
  output dir after every epoch — so if the output dir is in Drive, progress is
  safe even if the runtime dies mid-run. Restarting training creates a **new**
  `runs/<timestamp>/` folder; previous runs' checkpoints are NOT deleted.
- **Epochs vs. ceremony clock:** 8 epochs (the repo default) is ~4 hrs on a free
  T4; 3 epochs ~90 min. For a quick demo, 1 epoch (eval_loss ~0.62) is already a
  real fine-tune. Pick epochs to fit your time budget.
- **`load_best_model_at_end` is NOT enabled** — `train.py` saves the *final*
  epoch, which may not be the best (eval_loss often bottoms out mid-run on small
  data). Use the per-epoch checkpoints in Drive to export the best one.
- The full Colab cell auto-continues from training → export → Gradio launch, so
  finishing training prints the `*.gradio.live` link with no extra steps.
- **Public repo:** `trinitron88/ChinTranslator` (default branch `main`). The
  README's `ChinTranslator2` links are stale.

## How to use it going forward
- **Fresh training cycle:** open `ChinTranslator_V5_Colab.ipynb` in Colab, set
  GPU, adjust `EPOCHS`, Run all. Model lands in
  `MyDrive/ChinTranslator/runs/<timestamp>/whisper-cnh-turbo-ct2`.
- **Re-serve an existing model (no retrain):** mount Drive, then run
  `gradio_interface.py` with `CHIN_MODEL` set to the path in
  `MyDrive/ChinTranslator/LATEST_MODEL.txt`.
- **Emergency demo from a checkpoint:** run the one-liner above.

## Open follow-ups (not done yet)
- Optionally enable `load_best_model_at_end` (+ `metric_for_best_model`) in
  `train.py` so the best epoch is kept automatically.
- Consider lowering the default epochs, or surfacing an ETA warning.
