#!/usr/bin/env python3
"""
rescue_export.py — emergency path to a live demo WITHOUT retraining.

Finds the newest LoRA epoch-checkpoint already saved in Drive (across all runs),
merges it into the base model, converts to CT2, saves the result to Drive, and
launches the Gradio demo. Pure Python (no notebook magics) so it runs with a
single one-liner on a phone:

    !rm -rf /content/CT && \
      git clone -q -b claude/latest-result-5kTMp \
      https://github.com/trinitron88/ChinTranslator.git /content/CT && \
      python /content/CT/rescue_export.py

Drive must already be mounted in the session (the training cell does this). If
it isn't, run once in any cell:  from google.colab import drive; drive.mount('/content/drive')
"""

import glob
import os
import shutil
import subprocess
import sys


def _pip():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "ctranslate2>=4.4", "faster-whisper", "peft",
         "transformers>=4.44", "accelerate"],
        check=False,
    )


_pip()

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402

DRIVE_ROOT = "/content/drive/MyDrive/ChinTranslator"
BASE_ID = "openai/whisper-large-v3-turbo"
MERGED = "/content/whisper-cnh-turbo-merged"

if not os.path.isdir("/content/drive/MyDrive"):
    sys.exit("Drive not mounted. Run in a cell first:\n"
             "  from google.colab import drive; drive.mount('/content/drive')")

# Newest epoch checkpoint across ALL runs (finds run #1's epoch 1 even after a
# restart created a fresh run folder).
ckpts = glob.glob(f"{DRIVE_ROOT}/runs/*/whisper-cnh-turbo-lora/checkpoint-*")
if not ckpts:
    sys.exit(f"❌ No checkpoints found under {DRIVE_ROOT}/runs/*/whisper-cnh-turbo-lora/. "
             "No epoch has finished saving yet.")
ADAPTER = max(ckpts, key=os.path.getmtime)
RUN_DIR = ADAPTER.split("/whisper-cnh-turbo-lora/")[0]
CT2_OUT = f"{RUN_DIR}/whisper-cnh-turbo-ct2"
print(f"🔎 Using newest checkpoint: {ADAPTER}")

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print("🔗 Merging adapter into base ...")
base = WhisperForConditionalGeneration.from_pretrained(BASE_ID, torch_dtype=dtype)
PeftModel.from_pretrained(base, ADAPTER).merge_and_unload().save_pretrained(MERGED)
# Tokenizer/feature-extractor are unchanged from base; CT2 needs them alongside.
WhisperProcessor.from_pretrained(BASE_ID).save_pretrained(MERGED)

if os.path.exists(CT2_OUT):
    shutil.rmtree(CT2_OUT)
print(f"🔄 Converting → CT2 at {CT2_OUT} ...")
subprocess.run(
    ["ct2-transformers-converter", "--model", MERGED, "--output_dir", CT2_OUT,
     "--copy_files", "tokenizer.json", "preprocessor_config.json",
     "--quantization", "float16"],
    check=True,
)
with open(f"{DRIVE_ROOT}/LATEST_MODEL.txt", "w") as f:
    f.write(CT2_OUT + "\n")
print(f"✅ CT2 model saved to: {CT2_OUT}")

# Serve it (prints a public *.gradio.live link).
print("🚀 Launching Gradio ...")
os.environ["CHIN_MODEL"] = CT2_OUT
gradio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradio_interface.py")
subprocess.run([sys.executable, gradio], check=False)
