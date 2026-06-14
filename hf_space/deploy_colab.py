#!/usr/bin/env python3
"""
deploy_colab.py — push hf_space/ to the Hugging Face Space from Colab.

Run this from a Colab cell:

    %run /content/CT/hf_space/deploy_colab.py

or import and call deploy(). It self-heals the common fresh-session papercuts:
  * /content/CT missing or not a git repo  -> fresh clone
  * git remote pointing at the old (moved) URL -> repoint to REPO_URL
  * stray local changes in the checkout -> `reset --hard` instead of `pull`
  * HF_WRITE undefined -> read from env or Colab userdata, with a clear error

Every git command's output is printed, so a failure tells you *why* instead of
a bare non-zero exit.
"""

import os
import subprocess

# --- config ------------------------------------------------------------------
REPO_DIR = os.environ.get("CT_REPO_DIR", "/content/CT")
REPO_URL = "https://github.com/trinitron88/ChinTranslator.git"  # new location
# main is the trunk; deploy from it so the Space gets the current hf_space/ code
# (the CHIN_MODEL hard-fail + upload_model.py live here, not on stale realtime).
BRANCH = os.environ.get("CT_BRANCH", "main")
# Owner defaults to HF_USER (then "bsantisi"); a fresh Colab only needs the write
# token. Override the whole id with SPACE_REPO_ID if needed.
HF_USER = os.environ.get("HF_USER", "bsantisi")
SPACE_REPO_ID = os.environ.get("SPACE_REPO_ID", f"{HF_USER}/chin-realtime")


def _resolve_hf_token() -> str:
    """HF write token: explicit env first, then Colab secrets (userdata)."""
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
        "No HF write token found. Set HF_WRITE in the Colab cell "
        "(HF_WRITE = '...') or add it under the 🔑 Secrets panel."
    )


def _git(*args, check=True):
    r = subprocess.run(["git", "-C", REPO_DIR, *args],
                       capture_output=True, text=True)
    print(f"$ git {' '.join(args)}\n{r.stdout}{r.stderr}".rstrip(), flush=True)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)} exited {r.returncode}")
    return r


def _sync_repo():
    """Bring REPO_DIR to the tip of BRANCH, cloning fresh if needed."""
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        print(f"[deploy] {REPO_DIR} is not a git repo — cloning fresh", flush=True)
        subprocess.run(["rm", "-rf", REPO_DIR], check=True)
        subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, REPO_DIR],
                       check=True)
    else:
        _git("remote", "set-url", "origin", REPO_URL)  # follow the move
        _git("fetch", "origin")
        _git("checkout", BRANCH)
        _git("reset", "--hard", f"origin/{BRANCH}")  # no merge conflicts
    _git("log", "-1", "--oneline")  # confirm the deployed commit


def deploy():
    token = _resolve_hf_token()
    _sync_repo()
    from huggingface_hub import upload_folder
    upload_folder(
        folder_path=os.path.join(REPO_DIR, "hf_space"),
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        token=token,
    )
    print(f"[deploy] uploaded hf_space/ → {SPACE_REPO_ID} ✓", flush=True)


if __name__ == "__main__":
    deploy()
