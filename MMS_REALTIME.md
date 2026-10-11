# MMS Hakha Chin microphone demo

`mms_realtime.py` is a standalone, transcription-only experiment using Meta's
[`facebook/mms-1b-all`](https://huggingface.co/facebook/mms-1b-all) and its
Hakha Chin (`cnh`) language adapter. It listens to the browser microphone,
transcribes fixed chunks of 2–5 seconds (default: 3 seconds), and displays the
last 120 nonempty transcript rows. It does not translate to English or
generate speech.
The existing Whisper/V6 apps, training pipeline, and Space remain separate.

This is chunked CTC inference, not a continuously updating decoder. Words at
chunk boundaries can be cut, and decoding time adds to the chosen chunk
duration. An adjustable energy gate (default: −45 dBFS) skips quiet/silent or
very short chunks. This is not speech VAD: background noise can pass the gate,
and quiet speech can be skipped. No conversational accuracy result is available.
MMS was trained substantially on read religious speech; its support for `cnh`
does not establish accuracy on conversational Hakha Chin.

## Model license and attribution

The MMS model weights are provided by **Meta** under
[Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
Use this demo and those weights for noncommercial evaluation in accordance
with that license. The repository's other model and dependency licenses do
not remove MMS's noncommercial restriction. Retain the model attribution,
license link, and a description of changes when sharing a derived model.

Model: [Meta, `facebook/mms-1b-all`, model card and license](https://huggingface.co/facebook/mms-1b-all).
Research: Pratap et al., [*Scaling Speech Technology to 1,000+ Languages*](https://arxiv.org/abs/2305.13516), 2023.
This demo selects the existing `cnh` adapter; it does not train new weights.

## Where microphone audio goes

The browser sends microphone chunks to the Python backend that runs this
demo. When browser and backend are on your own computer, inference runs on
that computer. When the backend is Colab or another host, the audio leaves
your phone/computer and is processed there. With `--share`, the connection
also uses Gradio's public share tunnel; treat the backend operator and that
tunnel as part of the audio's path.

Microphone audio stays in browser/backend memory; this demo does not save
recordings to disk. **Clear / reset** stops capture, discards waiting browser
chunks, aborts the active request, and ignores late results. Server inference
that has already started may still finish, but its result does not repopulate
the cleared transcript.

Model files are downloaded from Hugging Face on first load. The demo uses
the downloaded model for inference rather than a hosted ASR API. Use
consented recordings, and tell speakers where the backend is running before
starting the microphone. A share URL exposes the demo to anyone who has the
URL, so use it only for a temporary test and stop the process afterward.

## Local setup: macOS or Linux

Use Python **3.11 or 3.12** in a dedicated checkout/environment. The commands
below use `python3.12`; substitute `python3.11` if that is your installed
version. `.venv` is already excluded by this repository's `.gitignore`.
This demo's PCM microphone recorder does not require ffmpeg.

```bash
git clone https://github.com/trinitron88/ChinTranslator.git
cd ChinTranslator
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Optional on CPU-only Linux: avoid downloading PyTorch's CUDA dependencies.
# Omit this line on macOS and when using a CUDA GPU.
# python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-mms.txt
python mms_realtime.py --device cpu
```

The macOS instructions target Apple Silicon. Intel macOS needs a separately
validated PyTorch stack or a Linux/Colab backend. CPU is the supported macOS
execution path for this demo; it does not select Apple's MPS backend.
On Linux with a compatible NVIDIA GPU/PyTorch
installation, choose `--device cuda`, or use `--device auto` to select CUDA
when available and CPU otherwise. `cuda` should fail visibly if CUDA is
unavailable. CPU inference on this 1-billion-parameter model can be slow;
measure it on your hardware before assuming it can keep up with live speech.

Internet access is needed for the initial model, processor, and `cnh` adapter
downloads. Allow sufficient disk space and memory for a large model. Later
loads can reuse the Hugging Face cache. Model-loading errors should identify
download/cache, dependency, adapter, or device problems instead of falling
back to a different model or language.

Open the printed local URL in a browser and click **Start microphone**. Grant
microphone permission and wait for a full chunk and its inference result.
**Stop** sends the final captured partial chunk and lets queued chunks finish;
quiet or very short chunks may still be skipped. **Clear / reset** discards
waiting audio and clears the transcript. Starting again begins a fresh
displayed session. Timestamp ranges describe received audio; decode latency
excludes capture, transport, and queue time.

The **Silence gate (dBFS)** number field accepts −60 through −20, default −45.
Lower it to accept quieter speech and more background noise. If processing
falls behind, **Stop** lets pending chunks finish. When a new chunk would
exceed eight waiting browser chunks, capture stops and reports that remaining
microphone audio was dropped; queued chunks still finish. Keep tests short
and check the status for skips, errors, and overload rather than assuming
continuous capture.

### Launch options

```bash
# Automatic CUDA/CPU selection; default 3-second chunks
python mms_realtime.py --device auto

# Explicit 5-second chunks (allowed range: 2 through 5 seconds)
python mms_realtime.py --device cpu --chunk-seconds 5

# Serve on another interface/port; HTTPS still needs a proxy or share URL
python mms_realtime.py --host 0.0.0.0 --port 7861

# Temporary public HTTPS URL for a phone or remote browser
python mms_realtime.py --device auto --share
```

`--share` is opt-in. `--host 0.0.0.0` makes the server reachable on available
network interfaces; it does not create HTTPS or authenticate users.

### Microphone permission and HTTPS

Browsers require a secure context for microphone access. `http://localhost`
and `http://127.0.0.1` normally qualify on the computer running the browser.
A plain HTTP LAN address such as `http://192.168.1.20:7860` usually does not.
For a phone, use the printed **HTTPS** share URL or your own HTTPS reverse
proxy, then allow microphone access in the browser. `localhost` on the phone
refers to the phone, not your laptop or Colab runtime.

If the microphone control is unavailable, first check the URL scheme,
browser permission, and selected input device. If text stops arriving, check
the backend output for inference errors and whether inference takes longer
than the selected chunk duration.

## Colab setup

Select a GPU runtime if available. Check that the runtime uses Python 3.11 or
3.12; an already-imported incompatible package may require a runtime restart
after installation. Run these cells in order:

```python
!git clone https://github.com/trinitron88/ChinTranslator.git /content/ChinTranslator
%cd /content/ChinTranslator
!python --version
!python -m pip install -r requirements-mms.txt
```

```python
!python mms_realtime.py --device auto --share
```

Wait for the model/adapter load and printed HTTPS share URL. Open that URL on
your phone or computer, grant microphone permission, and record there. The
backend runs in Colab; Colab disconnects and its model cache may disappear
when the runtime resets. Stop the running cell to end the demo. Browser
microphone chunks use HTTP(S); no WebRTC or TURN configuration is needed.

## Smoke test before opening the UI

```bash
python mms_realtime.py --device cpu --smoke-test

# Optional: a short consented recording, preferably PCM WAV
python mms_realtime.py --device auto --smoke-test --audio /path/to/clip.wav
```

The smoke test loads the actual model and `cnh` adapter, verifies that empty
input and silence are skipped, and performs a real forward pass on a
deterministic 440 Hz synthetic tone. That tone is **non-speech**: its verbatim
decoded text and decode latency establish execution, not recognition
accuracy. The JSON output includes the model revision, adapter/tokenizer
language, device, and those checks. An optional audio file is transcribed in
the selected fixed chunks without launching a browser. Use a native
speaker's reference transcript to assess that recording's correctness.
The first smoke test still downloads the model and needs the same memory as
the UI. A mocked unit test is not a successful model smoke test.

### Validation record

Verified on 2026-10-11 in a Linux x86-64 environment with Python 3.12.14,
PyTorch 2.6.0+cpu, Transformers 4.49.0, Gradio 5.23.1, NumPy 1.26.4,
SciPy 1.14.1, and SoundFile 0.13.1. The pinned environment resolves without
dependency conflicts. Actual `facebook/mms-1b-all` revision
`3d33597edbdaaba14a8e858e2c8caa76e3cec0cd` loaded on CPU, and both model
adapter and tokenizer reported `cnh` with matching vocabulary sizes.
Empty input and three seconds of silence returned empty text, skipped
inference, and reported zero decode time. A two-second synthetic 440 Hz tone
exercised real CTC inference and decoded to `""` in 2.35 seconds on the first
pass. That timing is a single diagnostic observation, not a speech benchmark
or evidence that CPU capture can keep up.
The final in-memory HTTP endpoint was also exercised with the real model:
48 kHz silence was skipped with empty text and zero inference time, and a
48 kHz two-second synthetic tone was resampled and decoded to `""` in 1.22
seconds. Neither signal is a native-speaker accuracy test.

All 24 offline regression tests pass (PCM/resampling, silence bypass, chunk
ordering/timestamps, partial flush, errors, transcript retention, and the
in-memory HTTP audio endpoint).
Headless Chromium was tested at a 390 × 844 phone viewport with a synthetic
microphone input and a fake decoder: Start/Stop, a short recording, two full
three-second windows plus a final partial window, ordered timestamps, and
Clear during an in-progress request all passed without JavaScript errors.
The short recording reached the backend exactly once (29,722 samples at
16 kHz); the longer recording produced 48,000 + 48,000 + 12,484 samples.
This verifies browser capture and control behavior, not speech recognition.

No consented native Hakha Chin recording,
reference transcript, exported V6 model, or GPU is available in this workspace;
speech accuracy, CER/WER, the paired comparison, and GPU behavior remain
unverified. Mac/Colab deployment and a real phone microphone also require
device testing. The prototype has not been validated for field use.

## Compare MMS with the exported V6 model

Use the **same consented native-speaker recordings**, not separate live
microphone sessions. Start with short, varied conversational turns, more than
one speaker, quiet and typical field conditions, and native-speaker-corrected
Chin references. Keep evaluation clips out of future training/mining unless
you explicitly retire them as an evaluation set. Record the speakers'
consent and permitted uses without putting personal details in a public PR.

1. Freeze a numbered set of WAVs and a UTF-8 reference manifest. For the
   snippets below, use mono 16 kHz PCM WAV. Give each clip an ID, file
   hash, duration, speaker pseudonym, and verbatim reference. Keep audio and
   references private unless speakers consented to publication.
2. Record the repository commit, Python/dependency versions, hardware,
   `facebook/mms-1b-all` revision/cache snapshot, adapter `cnh`, and the V6 CT2
   model directory/run ID. Use the **exported** `whisper-cnh-turbo-ct2` directory
   containing `model.bin`, not the LoRA adapter directory. Inspect its
   `chin_metadata.json` and preserve `language_token` (normally `id`) in
   Whisper inference. That is V6's trained surrogate token, not an MMS
   language code; MMS uses `cnh`.
3. Choose one fixed chunk size (for example 3 seconds) before scoring. For a
   demo comparison, split each WAV at those exact sample boundaries, including
   the final partial chunk, and feed **identical chunks** to both engines.
   Join each model's chunk transcripts with a single space in order. If you
   also test whole utterances, label that as a separate evaluation: whole-file
   recognition does not measure the chunked demo.
4. Use MMS greedy CTC decoding with the `cnh` processor/tokenizer and adapter,
   preserving its −45 dBFS energy gate and recording skipped chunks. For V6,
   use the transcription settings below, matching
   `evaluate_model.py`; do not invoke the batch UI's translation fallback,
   Google translation, TTS, or its different VAD/word-gap segmentation.
5. Save one row per original clip with `id`, `ref`, `mms`, and `v6`, plus the
   separate timing measurements. Check transcription errors blind to model
   identity with a native speaker before drawing conclusions.

Create `comparison-manifest.json` with one object per recording. Replace the
example reference with the native speaker's corrected transcript:

```json
[
  {"id": "speaker-a-001", "audio": "/private/recordings/a-001.wav", "ref": "Corrected Chin transcript goes here"}
]
```

From the repository directory, run this Python snippet in the MMS
environment. It loads once, uses the same public engine/preparation functions
as the demo, and writes a private results file. Change `CHUNK_SECONDS` only
before the evaluation starts:

```python
import hashlib
import json
from pathlib import Path
import soundfile as sf
from mms_realtime import MMSEngine, SAMPLE_RATE, prepare_audio

CHUNK_SECONDS = 3
THRESHOLD_DB = -45
engine = MMSEngine("auto")
manifest = json.loads(Path("comparison-manifest.json").read_text(encoding="utf-8"))
window = int(CHUNK_SECONDS * SAMPLE_RATE)
with Path("mms-results.jsonl").open("w", encoding="utf-8") as output:
    for clip in manifest:
        path = Path(clip["audio"])
        samples, rate = sf.read(path, dtype="float32", always_2d=False)
        if rate != SAMPLE_RATE or samples.ndim != 1:
            raise ValueError(f"Freeze mono 16 kHz WAV input first: {path}")
        if not len(samples):
            raise ValueError(f"Empty recording: {path}")
        chunks = []
        for start in range(0, len(samples), window):
            chunk = prepare_audio((rate, samples[start:start + window]))
            result = engine.transcribe(chunk, threshold_db=THRESHOLD_DB)
            chunks.append({"start_sample": start, "sample_count": len(chunk),
                           "text": result.text, "skipped": result.skipped,
                           "decode_seconds": result.latency})
        row = {**clip, "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "audio_seconds": len(samples) / SAMPLE_RATE,
               "chunk_seconds": CHUNK_SECONDS, "threshold_db": THRESHOLD_DB,
               "mms_revision": engine.model.config._commit_hash,
               "mms_device": engine.device, "mms_chunks": chunks,
               "mms": " ".join(c["text"] for c in chunks if c["text"])}
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
```

In the existing V6 environment, make sure `faster-whisper`, `numpy`, and
`soundfile` are installed and record their versions. Run the following from
the same directory. Set `V6_DIR` to that run's exported CT2 directory; set
`DEVICE` to match your intended hardware comparison. The snippet verifies
the WAV hashes and reuses the recorded sample boundaries:

```python
import hashlib
import json
import time
from pathlib import Path
import soundfile as sf
from faster_whisper import WhisperModel

V6_DIR = Path("/absolute/path/to/whisper-cnh-turbo-ct2")
DEVICE = "cpu"  # Use "cuda" for a compatible NVIDIA GPU.
language = json.loads((V6_DIR / "chin_metadata.json").read_text(
    encoding="utf-8")).get("language_token")
if not language:
    raise ValueError("Retrieve the trained V6 language token before comparing.")
model = WhisperModel(str(V6_DIR), device=DEVICE,
                     compute_type="float16" if DEVICE == "cuda" else "int8")
kwargs = dict(task="transcribe", language=language, beam_size=5,
              condition_on_previous_text=False, temperature=[0.0, 0.2, 0.4],
              compression_ratio_threshold=2.2, vad_filter=False)
rows = [json.loads(line) for line in Path("mms-results.jsonl").read_text(
    encoding="utf-8").splitlines() if line.strip()]
with Path("paired-results.jsonl").open("w", encoding="utf-8") as output:
    for row in rows:
        path = Path(row["audio"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["audio_sha256"]:
            raise ValueError(f"Recording changed since the MMS run: {path}")
        samples, rate = sf.read(path, dtype="float32", always_2d=False)
        if rate != 16000 or samples.ndim != 1:
            raise ValueError(f"Expected frozen mono 16 kHz WAV: {path}")
        chunks = []
        for boundary in row["mms_chunks"]:
            start = boundary["start_sample"]
            chunk = samples[start:start + boundary["sample_count"]]
            started = time.perf_counter()
            segments, _ = model.transcribe(chunk, **kwargs)
            text = " ".join(segment.text for segment in segments).strip()
            # Consume the one-shot segment generator inside the timed region.
            chunks.append({"text": text, "decode_seconds": time.perf_counter() - started})
        row.update(v6=" ".join(c["text"] for c in chunks if c["text"]),
                   v6_chunks=chunks, v6_model=str(V6_DIR.resolve()),
                   v6_device=DEVICE, v6_language_token=language)
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
```

Use the CUDA/float16 or CPU/int8 pair consistently and record which you used.
Do not pass MMS's `cnh` code as a Whisper language token. Missing V6 metadata
should stop the comparison; silently using Whisper auto-detection changes the
experiment. The snippets measure the demo's MMS energy-gate behavior and the
V6 evaluator's ungated transcription behavior; retain skipped chunks in the
report rather than dropping their reference text from the score.

For repeatable aggregate scores, save the paired rows as JSON Lines (one JSON
object per line) and run this in a separate scoring environment with `jiwer`
installed. Normalization matches `evaluate_model.py` and retains cnh
diacritics such as `ṭ`:

```python
import json
import re
from pathlib import Path
import jiwer

def normalize(text):
    text = re.sub(r"[^\w\s]", "", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

rows = [json.loads(line) for line in Path("paired-results.jsonl").read_text(
    encoding="utf-8").splitlines() if line.strip()]
refs = [row["ref"] for row in rows]
for name in ("mms", "v6"):
    hyps = [row[name] for row in rows]
    print(name, {
        "wer_raw_pct": 100 * jiwer.wer(refs, hyps),
        "wer_normalized_pct": 100 * jiwer.wer(
            list(map(normalize, refs)), list(map(normalize, hyps))),
        "cer_normalized_pct": 100 * jiwer.cer(
            list(map(normalize, refs)), list(map(normalize, hyps))),
    })
```

Measure model initialization/download time separately. The snippets record
decode times including the first inference; exclude the first actual
non-skipped inference when reporting steady-state timings, or warm up both
models explicitly first. `MMSEngine` synchronizes CUDA around its timed
inference; the V6 snippet fully consumes its segment generator before ending
the timer. Record audio duration and real-time factor
`inference_seconds / audio_seconds`. Report chunk duration, median and tail
inference times, skipped/failed chunks, and whether the processing queue grew.
Browser-to-visible-text delay includes chunk collection and transport too;
report it separately from model inference. Publish the protocol and results,
including failures, without asserting that one model wins before measuring.

`evaluate_model.py` remains the V6 held-out **Common Voice** evaluator:

```bash
python evaluate_model.py --model /absolute/path/to/whisper-cnh-turbo-ct2 \
    --data data/cv_cnh --split test --report v6-commonvoice-report.json
```

It accepts faster-whisper/CTranslate2 models, not `facebook/mms-1b-all`.
Its Common Voice report and the paired native-speaker field-recording report
cover different datasets and should remain clearly labeled.
