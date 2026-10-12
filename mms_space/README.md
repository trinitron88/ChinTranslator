---
title: Hakha Chin MMS speech-to-text
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.23.1
python_version: '3.12'
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# Hakha Chin transcription with an experimental English attempt

CPU Basic demo: Meta MMS `cnh` speech recognition, then an attempted English
translation using MADLAD-400 3B INT8 on the same server. No paid inference API,
new credentials, or hardware provisioning is used.

Record a short sample, then press **Stop**. The full retained Hakha Chin
transcript stays in the first box. Once pending audio finishes, the second box
translates the **latest four nonempty transcript chunks together**, not the whole
recording. Translation waits for Stop to avoid competing with active capture.
Use **Retry English** after a translation error. Clear resets both boxes;
new sessions discard stale translation responses. An already running server
translation may finish after cancellation, but its result is ignored.

Both speech recognition and translation are **unverified for conversational
Hakha Chin**. Fluent English can be wrong, particularly when the transcript is
wrong. Verify important content with a Hakha Chin speaker. This demo is for
noncommercial experimentation, not consequential decisions.

## Models and limits

- ASR: `facebook/mms-1b-all`, `cnh` adapter/tokenizer, float32 CPU.
  Meta MMS weights: **CC BY-NC 4.0**.
- Translation: Google MADLAD-400 3B, Apache 2.0; INT8 conversion by Heng-Shiou
  Sheu at `Heng666/madlad400-3b-mt-ct2-int8`, pinned to
  `b963b846ca4a5c106b7c64b863802599993e5f3b`.
- CTranslate2 CPU INT8, one worker, two threads, greedy decoding. Input at most
  1,200 characters / 256 tokens; output at most 128 tokens. Long input is rejected
  explicitly rather than silently truncated. Length-limited output is labeled.
- The first translation downloads and loads about 2.95 GB of weights. ASR stays
  available if translation fails. Concurrent translation requests return a busy
  message rather than building an unbounded server queue.

Audio and text travel over HTTPS to this Space. Inference runs here; no transcript
is sent to Google Translate or another inference provider. No audio/transcript
cache or recording file is written by the app. Ordinary hosting logs may remain.
Gradio analytics and Hub telemetry are disabled. Model downloads use public files
with no access token and do not execute repository code.

## Source and deployment

Source: https://github.com/trinitron88/ChinTranslator . The MMS entrypoint lives
in `mms_space/`; do not use the older Whisper `hf_space/deploy_colab.py`.

To export, copy `mms_space/app.py`, `mms_space/requirements.txt`, this README,
root `mms_realtime.py`, and root `mms_translation.py` into a Space's root. Select
CPU Basic explicitly. Preserve `ssr_mode=False` for the custom HTTP routes.
Do not provision or upgrade hardware automatically.

The browser recorder uses `/mms/transcribe`; translation uses `/mms/translate`.
The UI stays usable in both Gradio light and dark themes, including read-only
text and placeholders. Validation uses synthetic UI and endpoint samples;
those checks are not evidence of native-speaker accuracy.
