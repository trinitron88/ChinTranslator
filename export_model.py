#!/usr/bin/env python3
"""
export_model.py — turn the trained LoRA adapter into something faster-whisper
can serve.

train.py produces a LoRA *adapter* (whisper-cnh-turbo-lora/). The Gradio app
uses faster-whisper, which runs on CTranslate2 (CT2) and understands neither
PEFT adapters nor raw HF checkpoints. So two steps:

    1. MERGE  the adapter into the base weights  -> a standalone HF model
    2. CONVERT that HF model to CT2 format        -> what faster-whisper loads

Run AFTER training (needs the adapter to exist):
    !python export_model.py
    # then serve with:
    #   CHIN_MODEL=whisper-cnh-turbo-ct2 python gradio_interface.py

Note: merging needs a full-precision base (you cannot cleanly merge into an
8-bit base), so this loads the base in fp16/fp32 regardless of how it trained.
CT2 conversion of large-v3-turbo needs ctranslate2 >= 4.4.
"""

import subprocess
import sys


def _pip():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "transformers>=4.44", "peft", "accelerate",
         "ctranslate2>=4.4", "faster-whisper",
         "tiktoken", "sentencepiece"],
        check=False,
    )
    # PEFT's torchao guard raises on Colab's stale torchao 0.10 even though an
    # fp16 merge never uses torchao. Remove it so LoRA injection doesn't crash.
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
        check=False,
    )


_pip()

import argparse  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402

def _out_base():
    """Match train.py: persist to Drive on Colab, else current dir.

    Keeps the adapter/CT2 defaults pointed at the same Drive folder train.py
    writes to, so export finds the adapter and the served model survives a
    runtime reset. Explicit CLI args always win.
    """
    import os
    drive = "/content/drive/MyDrive/ChinTranslator"
    return drive if os.path.isdir("/content/drive/MyDrive") else "."


BASE_ID = "openai/whisper-large-v3-turbo"
_BASE = _out_base()
ADAPTER = f"{_BASE}/whisper-cnh-turbo-lora"   # where train.py saved it
MERGED = "whisper-cnh-turbo-merged"           # large intermediate; ok ephemeral
CT2 = f"{_BASE}/whisper-cnh-turbo-ct2"        # what faster-whisper serves


def merge(base_id, adapter, merged_out):
    print(f"\n🔗 Merging adapter '{adapter}' into base '{base_id}' ...")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = WhisperForConditionalGeneration.from_pretrained(base_id, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, adapter)
    model = model.merge_and_unload()  # fold LoRA deltas into the base weights
    model.save_pretrained(merged_out)
    # CT2 conversion needs the tokenizer + feature-extractor configs alongside.
    # Load them from the BASE model, not the adapter checkpoint: LoRA never
    # changes the tokenizer, and the base ships a complete prebuilt tokenizer,
    # whereas the training checkpoint only has partial tokenizer files (which
    # forces a slow->fast conversion that fails).
    WhisperProcessor.from_pretrained(base_id).save_pretrained(merged_out)
    print(f"✓ Merged model → {merged_out}/")


def to_ct2(merged_out, ct2_out, quantization):
    if Path(ct2_out).exists():
        shutil.rmtree(ct2_out)
    print(f"\n🔄 Converting → CT2 ({quantization}) ...")
    # ct2-transformers-converter is the documented path; --copy_files brings the
    # tokenizer/preprocessor that faster-whisper needs into the CT2 dir.
    cmd = [
        "ct2-transformers-converter",
        "--model", merged_out,
        "--output_dir", ct2_out,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
        "--quantization", quantization,
    ]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"✓ CT2 model → {ct2_out}/")


def main():
    ap = argparse.ArgumentParser(description="Merge LoRA + convert to CT2 for faster-whisper")
    ap.add_argument("--base", default=BASE_ID)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--merged-out", default=MERGED)
    ap.add_argument("--ct2-out", default=CT2)
    ap.add_argument("--quantization", default="float16",
                    help="CT2 quantization: float16 (GPU) / int8 (CPU) / int8_float16")
    ap.add_argument("--skip-merge", action="store_true",
                    help="merged dir already exists; only do the CT2 conversion")
    args = ap.parse_args()

    if not args.skip_merge:
        if not Path(args.adapter).exists():
            sys.exit(f"❌ adapter '{args.adapter}' not found — run train.py first.")
        merge(args.base, args.adapter, args.merged_out)

    to_ct2(args.merged_out, args.ct2_out, args.quantization)

    print("\n✅ Done. Serve it with:")
    print(f"   CHIN_MODEL={args.ct2_out} python gradio_interface.py")


if __name__ == "__main__":
    main()
