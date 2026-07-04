#!/usr/bin/env python3
"""
train.py — fine-tune whisper-large-v3-turbo on Hakha Chin (cnh) with LoRA + 8-bit.

Designed for a free Colab T4 (16 GB). The base model is frozen and loaded in
8-bit; only small LoRA adapters are trained. This fits in memory, trains in
minutes per epoch, and resists overfitting on a few hours of audio — full
fine-tuning a 0.8B model on this little data would not.

V6 changes over V5 (all aimed at conversational accuracy):

  * SURROGATE LANGUAGE TOKEN (--lang-token, default "id"). Whisper has no cnh
    token. V5 trained task-only, but faster-whisper always feeds a language
    token at inference — auto-detected, flapping between id/km/ms per utterance
    — a prompt the model never saw in training. V6 trains WITH one fixed token
    ("id" because that's where Whisper's detector already puts cnh speech) and
    the serving apps force the same token, so train and inference prompts
    match. The token rides along in chin_metadata.json.
  * Broader LoRA (--lora-targets full): all attention projections + MLP,
    not just q/v. ~3x the adapter params, still tiny vs the base.
  * SpecAugment on the log-mel features (built into HF Whisper) and on-the-fly
    waveform augmentation (gain / noise / speed) on the train split only —
    the data is clean read speech, the target is noisy conversation.
  * Best-checkpoint selection (eval loss) + early stopping, instead of
    keeping whatever the final epoch happened to be.

Pipeline:
    !python prepare_data.py        # builds data/cv_cnh/ (train/val/test)
    !python train.py               # writes whisper-cnh-turbo-lora/ (adapter)
    !python export_model.py        # merge + convert for faster-whisper

The output is a LoRA *adapter*, not a full model.
"""

import subprocess
import sys


def _pip():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "transformers>=4.46", "datasets>=4", "accelerate", "peft",
         "bitsandbytes", "evaluate", "jiwer", "librosa"],
        check=False,
    )


_pip()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Any, Dict, List, Union  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (  # noqa: E402
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

MODEL_ID = "openai/whisper-large-v3-turbo"
TASK = "transcribe"
LANG_TOKEN = "id"  # surrogate language token for cnh (see module docstring)
DATA_DIR = "data/cv_cnh"

LORA_PRESETS = {
    "attn": ["q_proj", "v_proj"],  # V5 behavior
    "full": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
}


def _out_base():
    """Persist to Drive when running on Colab, else the current dir."""
    drive = "/content/drive/MyDrive/ChinTranslator"
    return drive if os.path.isdir("/content/drive/MyDrive") else "."


OUTPUT_DIR = f"{_out_base()}/whisper-cnh-turbo-lora"


# ---------------------------------------------------------------------------
# Waveform augmentation (train split only, fresh every epoch)
# ---------------------------------------------------------------------------

def augment_wav(wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = wav
    if rng.random() < 0.3:  # speed/pitch perturb 0.9–1.1x via resampling
        rate = rng.uniform(0.9, 1.1)
        n = max(1, int(len(out) / rate))
        out = np.interp(np.linspace(0, len(out) - 1, n),
                        np.arange(len(out)), out).astype(np.float32)
    if rng.random() < 0.5:  # gain ±6 dB
        out = out * (10.0 ** (rng.uniform(-6, 6) / 20.0))
    if rng.random() < 0.5:  # gaussian noise at 8–30 dB SNR
        rms = float(np.sqrt(np.mean(out ** 2))) or 1e-4
        snr = rng.uniform(8, 30)
        noise_rms = rms / (10.0 ** (snr / 20.0))
        out = out + rng.normal(0.0, noise_rms, size=len(out)).astype(np.float32)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pads features + labels. Train examples arrive as raw waveforms (so
    augmentation is re-rolled every epoch); eval examples arrive with
    precomputed input_features. Both forms are handled here."""
    processor: Any
    augment: bool = False
    decoder_start_token_id: int = -1  # <|startoftranscript|>
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(1234))

    def __call__(self, features: List[Dict[str, Union[List, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = []
        for f in features:
            if "input_features" in f:
                input_features.append({"input_features": f["input_features"]})
            else:
                wav = np.asarray(f["audio"], dtype=np.float32)
                if self.augment:
                    wav = augment_wav(wav, self.rng)
                feats = self.processor.feature_extractor(
                    wav, sampling_rate=16000).input_features[0]
                input_features.append({"input_features": feats})
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # Cut the leading <|startoftranscript|>: the model's shift_tokens_right
        # re-prepends it as decoder_start_token_id. Without this cut, decoder
        # inputs become [SOT, SOT, lang, task, ...] — a doubled-SOT prompt that
        # inference (single SOT) never produces. NOTE: the classic recipe checks
        # tokenizer.bos_token_id here, but Whisper's bos_token is <|endoftext|>,
        # so that check never fires — compare against SOT itself.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def _normalize(text: str) -> str:
    """Light normalization for WER: case + punctuation, keep diacritics (ṭ)."""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser(description="LoRA fine-tune whisper-large-v3-turbo on cnh")
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--out", default=OUTPUT_DIR)
    ap.add_argument("--epochs", type=float, default=10.0,
                    help="max epochs; early stopping usually ends it sooner")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4,
                    help="1e-3 suits q/v-only LoRA; broader targets like it lower")
    ap.add_argument("--lang-token", default=LANG_TOKEN,
                    help="surrogate Whisper language token to train with and force "
                         "at inference (default: id). 'none' = V5 task-only mode")
    ap.add_argument("--lora-targets", choices=sorted(LORA_PRESETS), default="full")
    ap.add_argument("--no-augment", action="store_true", help="disable waveform augmentation")
    ap.add_argument("--no-specaugment", action="store_true", help="disable SpecAugment")
    ap.add_argument("--model", help="override base model id")
    ap.add_argument("--no-8bit", action="store_true", help="load base in full precision (no bitsandbytes)")
    ap.add_argument("--limit", type=int, help="use only N train / N//4 eval examples")
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU smoke test: tiny model, tiny data, 1 epoch — validates the pipeline, not the result")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model or ("openai/whisper-tiny" if args.dry_run else MODEL_ID)
    use_8bit = torch.cuda.is_available() and not args.no_8bit and not args.dry_run
    lang = None if args.lang_token == "none" else args.lang_token
    print(f"Device: {device}  |  base: {model_id}  |  8-bit: {use_8bit}  |  "
          f"lang-token: {lang or '(task-only)'}  |  lora: {args.lora_targets}")
    if device == "cpu" and not args.dry_run:
        print("⚠️  No GPU — real training needs CUDA. This will be a CPU/full-precision run.")

    # ---- data ----
    print(f"\n📂 Loading {args.data}")
    ds = load_from_disk(args.data)
    # V6 datasets have train/val/test; V5 datasets only train/test. Eval on val
    # when present so the official test split stays untouched for evaluate_model.py.
    eval_split = "val" if "val" in ds else "test"
    limit = args.limit or (8 if args.dry_run else None)
    if limit:
        ds["train"] = ds["train"].select(range(min(limit, len(ds["train"]))))
        ds[eval_split] = ds[eval_split].select(range(min(max(1, limit // 4), len(ds[eval_split]))))
    print(f"   train={len(ds['train'])}  eval({eval_split})={len(ds[eval_split])}")

    processor = WhisperProcessor.from_pretrained(model_id, language=lang, task=TASK)

    def tokenize_only(batch):
        # train split: keep the raw waveform; features are computed per-batch in
        # the collator so augmentation is different every epoch
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    def prepare_full(batch):
        batch["input_features"] = processor.feature_extractor(
            batch["audio"], sampling_rate=16000
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    train_ds = ds["train"].map(tokenize_only, remove_columns=["sentence"],
                               num_proc=1, desc="Tokenizing train")
    eval_ds = ds[eval_split].map(prepare_full, remove_columns=ds[eval_split].column_names,
                                 num_proc=1, desc="Extracting eval features")

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, augment=not args.no_augment,
        decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids(
            "<|startoftranscript|>"))

    # ---- model: (8-bit) frozen base + LoRA adapters ----
    print(f"\n🤖 Loading base ({'8-bit' if use_8bit else 'full precision'}) + attaching LoRA ...")
    if use_8bit:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        # with a frozen 8-bit base + gradient checkpointing, the encoder input
        # must be told to require grad so gradients reach the LoRA layers
        model.model.encoder.conv1.register_forward_hook(
            lambda module, inp, out: out.requires_grad_(True)
        )
    else:
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        model.to(device)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.forced_decoder_ids = None

    if not args.no_specaugment:
        # HF Whisper's built-in SpecAugment (applies only while training)
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
        model.config.mask_time_length = 10
        model.config.mask_feature_prob = 0.05
        model.config.mask_feature_length = 10

    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        target_modules=LORA_PRESETS[args.lora_targets],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ---- train ----
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=50,
        num_train_epochs=1.0 if args.dry_run else args.epochs,
        fp16=torch.cuda.is_available(),
        per_device_eval_batch_size=args.batch,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,  # keep Drive from filling with per-epoch checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        report_to=["tensorboard"],
        # PEFT + 8-bit: generation-in-eval is unreliable, so we eval on loss and
        # measure WER separately after training. These two keep the Trainer from
        # dropping the LoRA-required columns / mislabeling inputs.
        predict_with_generate=False,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # Write the serving metadata BEFORE training starts (and again with the
    # final save): rescue_export.py builds demos from mid-run epoch
    # checkpoints, and it needs the surrogate language token to serve them
    # with the right decoder prompt.
    os.makedirs(args.out, exist_ok=True)
    meta = {"language_token": lang, "task": TASK, "base_model": model_id,
            "lora_targets": args.lora_targets}
    with open(os.path.join(args.out, "chin_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    model.config.use_cache = False

    print("\n🚀 Training ...")
    trainer.train()

    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"\n✅ Adapter saved → {args.out} (best checkpoint by eval loss)")

    # ---- post-hoc WER on the held-out official test set ----
    try:
        import evaluate
        wer_metric = evaluate.load("wer")
        test_split = "test" if "test" in ds else eval_split
        test_ds = ds[test_split]
        if limit:
            test_ds = test_ds.select(range(min(max(1, limit // 4), len(test_ds))))
        model.eval()
        infer_device = next(model.parameters()).device
        gen_dtype = torch.float16 if infer_device.type == "cuda" else torch.float32
        gen_kwargs = {"max_new_tokens": 225}
        if lang:
            gen_kwargs.update(language=lang, task=TASK)
        preds, refs = [], []
        for ex in test_ds:
            feats = processor.feature_extractor(
                np.asarray(ex["audio"], dtype=np.float32), sampling_rate=16000,
                return_tensors="pt").input_features.to(infer_device).to(gen_dtype)
            with torch.no_grad():
                ids = model.generate(input_features=feats, **gen_kwargs)
            preds.append(processor.tokenizer.decode(ids[0], skip_special_tokens=True))
            refs.append(ex["sentence"])
        raw = 100 * wer_metric.compute(predictions=preds, references=refs)
        norm = 100 * wer_metric.compute(predictions=[_normalize(p) for p in preds],
                                        references=[_normalize(r) for r in refs])
        print(f"\n📈 WER on {test_split} ({len(refs)} clips): "
              f"{raw:.1f}% raw / {norm:.1f}% normalized")
    except Exception as e:  # noqa: BLE001
        print(f"(WER eval skipped: {e})")


if __name__ == "__main__":
    main()
