# V6 — Accuracy push for conversational Hakha Chin

Everything in this round targets one gap: V5 trained on clean, read,
32-speaker audio, but the goal is understanding **real conversation** —
different voices, noise, natural speech. Changes land in five areas: a
train/inference prompt bug, more data, robustness training, better checkpoint
selection, and honest measurement.

**The bar:** the only published result on the Common Voice `cnh` test split is
**31.4% WER** (`gchhablani/wav2vec2-large-xlsr-cnh`). No published Whisper
fine-tune for Hakha Chin exists — this project is first. Every training run
now reports WER on that exact split, so we know where we stand.

## 1. Fixed: train/inference prompt mismatch (surrogate language token)

Whisper has no `cnh` language token, so V5 trained "task-only" — labels began
`<|sot|><|transcribe|>`. But **faster-whisper always inserts a language token
at inference**, auto-detected per utterance (cnh speech flaps between
`id`/`km`/`ms`). The served model was being prompted with a shape it never saw
in training, differently from one utterance to the next.

A second, subtler instance of the same disease: the classic fine-tune recipe's
data collator tries to strip the leading `<|startoftranscript|>` from labels
(the model re-prepends it), but checks `tokenizer.bos_token_id` — which for
Whisper is `<|endoftext|>`, so the strip never fired and V5 trained on
doubled-SOT decoder prompts (`[SOT, SOT, lang, ...]`) that inference (single
SOT) never produces. V6's collator compares against the actual SOT token.

V6 trains with a fixed stand-in token — default `id`, since that's where
Whisper's detector already places cnh speech — and every serving app forces
the same token. The token travels in `chin_metadata.json`: train.py writes it
into the adapter, export_model.py copies it into the CT2 dir,
upload_model.py ships it to the HF repo, and all three apps
(gradio_interface, realtime, hf_space) read it automatically. V5-era models
have no metadata file and keep the old auto-detect behavior. Env `CHIN_LANG`
overrides everywhere (`CHIN_LANG=""` forces auto-detect).

## 2. More data, more voices

What Common Voice 17 `cnh` actually contains (V5 used only train+dev):

| pool | clips | hours | speakers | V5 | V6 |
|------|------:|------:|---------:|----|----|
| train | 817 | 0.65 | **6** | 80% of it | all of it |
| dev | 761 | 0.76 | 26 | 80% of it | all of it |
| test | 763 | 0.86 | **158** | unused | **held-out eval** |
| other (unvalidated) | 3,292 | 3.18 | **209** | unused | vote-filter + mining |
| Bible archive (V4) | 540 segs | ~0.4 | 1 | retired | optional mix-in |

- **No more 20% throwaway.** V5 pooled train+dev and re-split 80/20; the 20%
  duplicated what the official test split already provides. V6 trains on all
  of train+dev, carves only a small 8% `val` set for early stopping, and
  evaluates on the official test split — 158 speakers the model never heard,
  which is exactly the "new voices at a family gathering" condition.
- **The `other` pool** is clips that never got enough review votes — not
  rejected, just unreviewed. It has more audio than everything else combined
  and the best gender balance (815 female-voiced clips). Two ways in:
  - `--include-other votes` (default): clips with ≥1 upvote and no net
    downvotes (~600 clips).
  - **`mine_other.py`** (new): transcribe the whole pool with your current
    model and keep clips where the model agrees with the label (CER ≤ 15%).
    If an independent model reads the audio and reproduces the label, the
    label is almost certainly right. This is the data flywheel:
    train → mine → retrain → mine again with the better model.
- **Bible mix-in** (optional): `prepare_data.py --bible-json
  archive/bible_dataset/aligned_training_data.json --bible-audio-dir
  <Drive>/Audio` adds the V4 alignment — read male speech, but real cnh
  phonology the acoustic model can learn from.

## 3. Robustness: augmentation on clean read speech

Conversational audio is noisy, variably loud, variably paced. Two layers,
both on by default in train.py, train split only:

- **Waveform augmentation**, re-rolled fresh every epoch (the train split now
  stores raw audio and features are computed per-batch): random gain ±6 dB,
  gaussian noise at 8–30 dB SNR, speed/pitch perturbation 0.9–1.1×.
- **SpecAugment** (built into HF Whisper): random time/frequency masks on the
  log-mel features.

Disable with `--no-augment` / `--no-specaugment` to A/B.

## 4. Better fine-tune mechanics

- **LoRA on all attention projections + MLP** (`q,k,v,out,fc1,fc2`) instead of
  q/v only — `--lora-targets attn` restores V5. Default LR drops 1e-3 → 5e-4
  to match the broader adapter.
- **Best-checkpoint selection** (`load_best_model_at_end` on eval loss) +
  **early stopping** (patience 3). V5 kept whatever the final epoch was, and
  eval loss usually bottomed out mid-run. `save_total_limit=3` keeps Drive
  from filling up.
- Default max epochs 10; early stopping typically ends the run sooner.

## 5. Measurement: evaluate_model.py

Runs the same faster-whisper engine the apps serve with, against the held-out
official test split, and prints raw + normalized WER, CER, and the worst-5
utterances (fastest way to see *what kind* of errors remain):

```bash
python evaluate_model.py --model <run>/whisper-cnh-turbo-ct2
python evaluate_model.py --model <run>/whisper-cnh-turbo-ct2 --model large-v3-turbo   # compare
```

The Colab notebook now runs this automatically after export and writes
`eval_report.json` into the run folder.

## Suggested first V6 cycle

1. Open `ChinTranslator_V5_Colab.ipynb` (now V6), Run all — defaults do:
   full train+dev + vote-filtered other, lang token, augmentation, early
   stopping, eval on held-out test.
2. Note the WER. Then run the flywheel:
   ```bash
   python mine_other.py --model <run>/whisper-cnh-turbo-ct2
   python prepare_data.py --other-keep other_keep.json
   python train.py && python export_model.py && python evaluate_model.py --model ...
   ```
3. Compare `eval_report.json` between runs; keep what wins.

## Ideas backlog (researched, not yet built)

Ordered by expected impact per effort:

1. **CMU Wilderness CNHBSM — 21.7 hours of aligned cnh speech** (Hakha Chin
   New Testament, 12,285 utterances, festvox.org/cmu_wilderness/CNHBSM). An
   order of magnitude more audio than everything we train on now. Read
   Bible speech, but at this data scale that usually still lifts
   conversational WER substantially. Needs rebuilding from bible.is via
   github.com/festvox/datasets-CMU_Wilderness (research-use license caution).
2. **Synthetic speech from text via `facebook/mms-tts-cnh`.** Meta ships a
   Hakha Chin TTS voice. Feed it cnh text (e.g. `HuggingFaceFW/fineweb-2`
   cnh subset, or the unused `validated_sentences.tsv` in Common Voice) to
   mint unlimited (audio, transcript) pairs with vocabulary far beyond the
   current 2.4k sentences. Mix at ~20–30% of training data.
3. **Newer Common Voice via Mozilla Data Collective** — cnh is at ~6h
   recorded / 300 speakers in CV v24 (vs 2.4h validated in v17 mirrors).
   Needs an account + download; also the single highest-leverage *community*
   action is getting Chin speakers to validate clips at
   commonvoice.mozilla.org — validation, not recording, is the bottleneck.
4. **Field-recording flywheel** — record real conversations (with consent),
   transcribe with the current model, have a native speaker correct, add to
   training. Even 1–2 hours of true conversational cnh would attack the
   domain gap directly, which no amount of read speech can.
5. **Translation upgrade**: Google Translate has *official* cnh NMT support
   since June 2024 (the free endpoint the apps call already benefits). For
   offline/on-device translation, `google/madlad400-3b-mt` supports cnh
   (`<2en>` prefix); `wayne-cc/hakha-chin-to-english-student-translator` is a
   tiny 0.2B Marian distillation worth benchmarking. NLLB does **not**
   support cnh (nearest is Mizo, not mutually intelligible).
6. **MMS-1B-all as a second opinion** — Meta's MMS ASR supports cnh (plus
   Falam/Tedim). CTC decoding, Bible-domain training. Could ensemble or
   cross-check ("both models agree" → high confidence in the realtime UI).
7. **Chunk length / VAD tuning for conversation** — Common Voice clips are
   ~3s; real conversational turns are longer with pauses. The realtime VAD
   settings (silence 500ms, pad 140ms) are worth field-tuning.
