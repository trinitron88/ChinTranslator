#!/usr/bin/env python3
"""
upload_model.py — push the fine-tuned CT2 model to a Hugging Face MODEL repo so
the Space can serve it.

This is the step that was missing and the reason the Space silently fell back to
stock Whisper: the Space runs on HF infrastructure and CANNOT see your Google
Drive, so the Space's `CHIN_MODEL` variable must be an HF *model-repo id*, not a
Drive path. `export_model.py` writes the CT2 model to Drive; this uploads that
folder to a model repo, then you point the Space at the printed repo id.

Run from Colab (Drive mounted) AFTER export_model.py has produced the CT2 dir:

    %run /content/CT/hf_space/upload_model.py
    # or, explicitly:
    python hf_space/upload_model.py \
        --model-dir /content/drive/MyDrive/ChinTranslator/model_v5/whisper-cnh-turbo-ct2 \
        --repo bsantisi/whisper-cnh-turbo-ct2

Then: Space → Settings → Variables → CHIN_MODEL = <the printed repo id>, and
restart the Space.
"""

import argparse
import os

# Defaults match export_model.py's Drive conventions (model_v5/ per session memory).
DEFAULT_MODEL_DIR = "/content/drive/MyDrive/ChinTranslator/model_v5/whisper-cnh-turbo-ct2"
DEFAULT_REPO = "bsantisi/whisper-cnh-turbo-ct2"


def _resolve_hf_token() -> str:
    """HF write token: explicit env first, then Colab secrets (userdata).

    Mirrors hf_space/deploy_colab.py so both scripts find the token the same way.
    """
    tok = os.environ.get("HF_WRITE") or os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from google.colab import userdata  # type: ignore
        for key in ("HF_WRITE", "HF_TOKEN"):
            try:
                val = userdata.get(key)
                if val:
                    return val
            except Exception:
                pass
    except Exception:
        pass
    raise RuntimeError(
        "No HF write token found. Set HF_WRITE in the cell (HF_WRITE = '...') "
        "or add it under the 🔑 Secrets panel."
    )


def main():
    ap = argparse.ArgumentParser(description="Upload the CT2 model to an HF model repo")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                    help="local CT2 dir produced by export_model.py")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="target HF model-repo id (user/name)")
    ap.add_argument("--private", action="store_true",
                    help="create the model repo as private (then the Space needs an HF_TOKEN secret)")
    args = ap.parse_args()

    if not os.path.isdir(args.model_dir):
        raise SystemExit(
            f"❌ '{args.model_dir}' not found. Run export_model.py first to produce "
            f"the CT2 model (it converts the LoRA adapter → CTranslate2)."
        )
    # A faster-whisper CT2 model dir has model.bin + config.json at minimum; warn
    # (don't hard-fail) so an unusual-but-valid layout can still be uploaded.
    missing = [f for f in ("model.bin", "config.json")
               if not os.path.exists(os.path.join(args.model_dir, f))]
    if missing:
        print(f"⚠️  {args.model_dir} is missing {missing} — is this really a CT2 "
              f"export? Uploading anyway.")

    from huggingface_hub import HfApi
    api = HfApi(token=_resolve_hf_token())
    print(f"↑ creating/ensuring model repo {args.repo} (private={args.private}) ...")
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    print(f"↑ uploading {args.model_dir} → {args.repo} ...")
    api.upload_folder(folder_path=args.model_dir, repo_id=args.repo, repo_type="model")

    print(f"\n✓ uploaded → https://huggingface.co/{args.repo}")
    print("\nNext steps on the Space:")
    print(f"  • Settings → Variables → CHIN_MODEL = {args.repo}")
    if args.private:
        print("  • Settings → Secrets   → HF_TOKEN  = <a read token>  (repo is private)")
    print("  • Restart the Space. The log should show:")
    print(f"      Loading model: {args.repo} on cuda")


if __name__ == "__main__":
    main()
