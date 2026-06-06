#!/usr/bin/env python3
"""
train.py — fine-tune whisper-large-v3-turbo on Hakha Chin (cnh) with LoRA + 8-bit.

Designed for a free Colab T4 (16 GB). The base model is frozen and loaded in
8-bit; only small LoRA adapters are trained. This fits in memory, trains in
minutes, and resists overfitting on a ~1.3k-clip dataset — full fine-tuning a
0.8B model on this little data would not.

Pipeline:
    !python prepare_data.py        # builds data/cv_cnh/
    !python train.py               # writes whisper-cnh-turbo-lora/ (adapter)

The output is a LoRA *adapter*, not a full model. To serve it, load the base
model + adapter, or merge (see the note at the bottom of this file).
"""

import subprocess
import sys


def _pip():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "transformers>=4.44", "datasets>=4", "accelerate", "peft",
         "bitsandbytes", "evaluate", "jiwer", "librosa"],
        check=False,
    )


_pip()

import argparse  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any, Dict, List, Union  # noqa: E402

import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (  # noqa: E402
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

MODEL_ID = "openai/whisper-large-v3-turbo"
TASK = "transcribe"  # cnh has no Whisper language token; train task-only
DATA_DIR = "data/cv_cnh"


def _out_base():
    """Persist to Drive when running on Colab, else the current dir.

    Outputs were landing in ephemeral /content on Colab and vanishing on
    runtime reset. If Drive is mounted, default the adapter into the project
    folder there so a fresh cycle survives. Explicit --out always wins.
    """
    import os
    drive = "/content/drive/MyDrive/ChinTranslator"
    return drive if os.path.isdir("/content/drive/MyDrive") else "."


OUTPUT_DIR = f"{_out_base()}/whisper-cnh-turbo-lora"


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # cut the BOS the tokenizer prepends; the model adds it during generation
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main():
    ap = argparse.ArgumentParser(description="LoRA fine-tune whisper-large-v3-turbo on cnh")
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--out", default=OUTPUT_DIR)
    ap.add_argument("--epochs", type=float, default=8.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)  # LoRA likes a higher LR
    ap.add_argument("--model", help="override base model id")
    ap.add_argument("--no-8bit", action="store_true", help="load base in full precision (no bitsandbytes)")
    ap.add_argument("--limit", type=int, help="use only N train / N//4 test examples")
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU smoke test: tiny model, tiny data, 1 epoch — validates the pipeline, not the result")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model or ("openai/whisper-tiny" if args.dry_run else MODEL_ID)
    # 8-bit needs CUDA + bitsandbytes; auto-fall back so the script is CPU-runnable
    use_8bit = torch.cuda.is_available() and not args.no_8bit and not args.dry_run
    print(f"Device: {device}  |  base: {model_id}  |  8-bit: {use_8bit}")
    if device == "cpu" and not args.dry_run:
        print("⚠️  No GPU — real training needs CUDA. This will be a CPU/full-precision run.")

    # ---- data ----
    print(f"\n📂 Loading {args.data}")
    ds = load_from_disk(args.data)
    limit = args.limit or (8 if args.dry_run else None)
    if limit:
        ds["train"] = ds["train"].select(range(min(limit, len(ds["train"]))))
        ds["test"] = ds["test"].select(range(min(max(1, limit // 4), len(ds["test"]))))
    print(f"   train={len(ds['train'])}  test={len(ds['test'])}")

    processor = WhisperProcessor.from_pretrained(model_id, task=TASK)

    def prepare(batch):
        # "audio" is a 16 kHz float32 array (see prepare_data.py)
        batch["input_features"] = processor.feature_extractor(
            batch["audio"], sampling_rate=16000
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    ds = ds.map(prepare, remove_columns=ds["train"].column_names, num_proc=1,
                desc="Extracting features")

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

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

    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "v_proj"],
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
        logging_steps=25,
        report_to=["tensorboard"],
        # PEFT + 8-bit: generation-in-eval is unreliable, so we eval on loss and
        # measure WER separately after training. These two keep the Trainer from
        # dropping the LoRA-required columns / mislabeling inputs.
        predict_with_generate=False,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collator,
        processing_class=processor.feature_extractor,
    )
    model.config.use_cache = False

    print("\n🚀 Training ...")
    trainer.train()

    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"\n✅ Adapter saved → {args.out}")

    # ---- post-hoc WER on the eval set (generation works fine outside Trainer) ----
    try:
        import evaluate
        wer = evaluate.load("wer")
        model.eval()
        infer_device = next(model.parameters()).device  # Trainer may have moved it (cuda/mps)
        gen_dtype = torch.float16 if infer_device.type == "cuda" else torch.float32
        preds, refs = [], []
        for ex in ds["test"]:
            feats = torch.tensor(ex["input_features"]).unsqueeze(0).to(infer_device).to(gen_dtype)
            with torch.no_grad():
                ids = model.generate(input_features=feats, max_new_tokens=225)
            preds.append(processor.tokenizer.decode(ids[0], skip_special_tokens=True))
            labels = [t for t in ex["labels"] if t != -100]
            refs.append(processor.tokenizer.decode(labels, skip_special_tokens=True))
        print(f"\n📈 Eval WER: {100 * wer.compute(predictions=preds, references=refs):.1f}%")
    except Exception as e:  # noqa: BLE001
        print(f"(WER eval skipped: {e})")


# ---------------------------------------------------------------------------
# To serve the result with plain transformers:
#     from peft import PeftModel
#     base = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
#     model = PeftModel.from_pretrained(base, "whisper-cnh-turbo-lora")
#     model = model.merge_and_unload()          # fold adapter into base weights
#     model.save_pretrained("whisper-cnh-turbo-merged")
# The Gradio app uses faster-whisper (CTranslate2); to use it there, convert the
# merged model with: ct2-transformers-converter --model whisper-cnh-turbo-merged ...
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
