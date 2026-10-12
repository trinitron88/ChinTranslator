#!/usr/bin/env python3
"""Isolated, chunked Hakha Chin ASR with Meta MMS. See MMS_REALTIME.md."""

import argparse
import json
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import resample_poly

MODEL_ID = "facebook/mms-1b-all"
LANGUAGE = "cnh"
SAMPLE_RATE = 16000
MIN_AUDIO_SECONDS = 0.25
MAX_TRANSCRIPT_ROWS = 120


def prepare_audio(audio):
    """Gradio (sample rate, samples) -> finite mono float32 at 16 kHz."""
    if audio is None:
        return np.empty(0, dtype=np.float32)
    sample_rate, samples = audio
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("Audio sample rate must be a positive integer.")
    samples = np.asarray(samples)
    if samples.size == 0:
        return np.empty(0, dtype=np.float32)
    if samples.ndim not in (1, 2) or (samples.ndim == 2 and samples.shape[1] > 8):
        raise ValueError("Expected mono or samples-by-channels audio.")
    if np.issubdtype(samples.dtype, np.integer):
        limits = np.iinfo(samples.dtype)
        if limits.min == 0:
            midpoint = (limits.max + 1) / 2
            samples = (samples.astype(np.float32) - midpoint) / midpoint
        else:
            samples = samples.astype(np.float32) / max(abs(limits.min), limits.max)
    elif np.issubdtype(samples.dtype, np.floating):
        samples = samples.astype(np.float32)
    else:
        raise ValueError("Audio samples must be integer PCM or floating point.")
    if not np.isfinite(samples).all():
        raise ValueError("Audio contains NaN or infinite values.")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    samples = np.clip(samples, -1, 1)
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
        samples = resample_poly(samples, SAMPLE_RATE // divisor, sample_rate // divisor)
    return np.asarray(samples, dtype=np.float32)


def has_signal(samples, threshold_db=-45.0):
    """Energy gate, not speech classification: require 100 ms above threshold."""
    if len(samples) < int(MIN_AUDIO_SECONDS * SAMPLE_RATE):
        return False
    centered = samples - samples.mean()
    frame_size = SAMPLE_RATE // 50  # 20 ms frames
    usable = len(centered) // frame_size * frame_size
    frames = centered[:usable].reshape(-1, frame_size)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return np.count_nonzero(rms >= 10 ** (threshold_db / 20)) >= 5


@dataclass
class Result:
    text: str = ""
    latency: float = 0.0
    skipped: bool = False


class MMSEngine:
    """Load model/tokenizer and the cnh adapter once; serialize model access."""

    def __init__(self, device="auto"):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch or use --device cpu.")
        self.device = device
        self.torch = torch
        self.lock = threading.Lock()
        # Too many CPU threads can make these short chunks substantially slower.
        if device == "cpu":
            torch.set_num_threads(min(torch.get_num_threads(), 4))
        print(f"Loading {MODEL_ID} and {LANGUAGE} adapter on {device}...", flush=True)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.processor.tokenizer.set_target_lang(LANGUAGE)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
        self.model.load_adapter(LANGUAGE)
        self.model.to(device=device, dtype=torch.float32).eval()
        if self.model.config.vocab_size != len(self.processor.tokenizer):
            raise RuntimeError("MMS adapter/tokenizer vocabulary mismatch; refusing to transcribe.")
        print(f"Ready: adapter={self.model.target_lang}, tokenizer={self.processor.tokenizer.target_lang}, "
              f"revision={self.model.config._commit_hash}", flush=True)

    def transcribe(self, samples, threshold_db=-45.0):
        if not has_signal(samples, threshold_db):
            return Result(skipped=True)
        with self.lock, self.torch.inference_mode():
            if self.device == "cuda":
                self.torch.cuda.synchronize()
            started = time.perf_counter()
            inputs = self.processor(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            logits = self.model(**inputs).logits
            token_ids = logits.argmax(dim=-1)
            text = self.processor.batch_decode(token_ids)[0].strip()
            if self.device == "cuda":
                self.torch.cuda.synchronize()
            return Result(text=text, latency=time.perf_counter() - started)


@dataclass
class Session:
    pending: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    seconds: float = 0.0
    rows: list = field(default_factory=list)
    running: bool = False


def timestamp(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}:{seconds:05.2f}"


def consume(session, samples, engine, chunk_seconds, threshold_db, flush=False):
    """Consume received samples in order; timestamps refer to received audio."""
    session.pending = np.concatenate((session.pending, samples))
    window = int(chunk_seconds * SAMPLE_RATE)
    statuses = []
    while len(session.pending) >= window or (flush and len(session.pending)):
        count = min(window, len(session.pending))
        chunk, session.pending = session.pending[:count], session.pending[count:]
        start = session.seconds
        duration = count / SAMPLE_RATE
        session.seconds += duration
        try:
            result = engine.transcribe(chunk, threshold_db)
        except Exception as exc:
            statuses.append(f"Error at {timestamp(start)}: {type(exc).__name__}: {exc}")
            continue
        if result.skipped:
            statuses.append("Silent/quiet or very short chunk skipped; no inference.")
        elif result.text:
            session.rows.append(f"[{timestamp(start)}–{timestamp(session.seconds)}] {result.text}\n"
                                f"  Decode latency: {result.latency:.2f} s; audio: {duration:.2f} s")
            session.rows = session.rows[-MAX_TRANSCRIPT_ROWS:]
            status = f"Decoded {duration:.2f} s of audio in {result.latency:.2f} s."
            if result.latency > duration:
                status += " Processing is slower than capture; stop the mic to let queued chunks finish."
            statuses.append(status)
        else:
            statuses.append(f"No text decoded ({result.latency:.2f} s inference).")
    return "\n\n".join(session.rows), " ".join(statuses) or "Listening; collecting a full chunk..."


RECORDER_JS = r"""() => {
  const bind = () => {
    const start = document.getElementById('mms-start');
    if (!start) { requestAnimationFrame(bind); return; }
    if (start.dataset.english === 'true' && !document.getElementById('mms-english')) { requestAnimationFrame(bind); return; }
    if (start.dataset.bound) return;
    start.dataset.bound = 'true';
    const stop = document.getElementById('mms-stop');
    const clear = document.getElementById('mms-clear');
    const status = document.getElementById('mms-status');
    const transcript = document.getElementById('mms-transcript');
    const threshold = document.getElementById('mms-threshold');
    const seconds = Number(start.dataset.seconds);
    let context, source, processor, muted, microphone, recording = false;
    let pending, filled = 0, received = 0, generation = 0, rows = [];
    let queue = [], sending = false, controller, captureNotice = '';
    // ASR and translation drain independently, one request at a time each.
    const english = document.getElementById('mms-english');
    const englishStatus = document.getElementById('mms-english-status');
    const translateButton = document.getElementById('mms-translate');
    let translationGeneration = 0, translationController, translationSending = false;
    let translationQueue = [], englishRows = [];
    const maxWaitingTranslations = 8;
    function renderEnglish() {
      if (!english) return;
      english.value = englishRows.map(row => {
        let text = row.text;
        if (row.state === 'queued') text = 'Waiting for English translation…';
        if (row.state === 'translating') text = 'Translating to English…';
        if (row.state === 'failed') text = `Translation failed: ${row.error}`;
        return `[${stamp(row.start)}–${stamp(row.end)}] ${text}`;
      }).join('\n\n');
      english.scrollTop = english.scrollHeight;
      const failed = englishRows.filter(row => row.state === 'failed').length;
      englishStatus.textContent = translationSending ?
        `Translating on CPU; ${translationQueue.length} chunks waiting. First use loads the model…` :
        (recording || sending ? 'English appears as each transcription chunk is translated.' :
          (englishRows.length ? 'English translation queue finished.' : 'Ready for live English translation.'));
      if (failed) englishStatus.textContent += ` ${failed} failed chunks; use Retry English. Original transcript preserved.`;
      translateButton.disabled = translationSending || translationQueue.length > 0 || !failed;
    }
    function resetEnglish() {
      ++translationGeneration;
      if (translationController) translationController.abort();
      translationController = null; translationSending = false;
      translationQueue = []; englishRows = [];
      renderEnglish();
    }
    async function drainEnglish() {
      if (!english || translationSending || !translationQueue.length) return;
      const version = translationGeneration;
      translationSending = true;
      try {
        while (version === translationGeneration && translationQueue.length) {
          const row = translationQueue.shift();
          const pendingTranslation = new AbortController();
          translationController = pendingTranslation;
          row.state = 'translating';
          renderEnglish();
          try {
            // Another browser (or cancelled server inference) can hold the
            // translator lock. Retry briefly, then leave this row retryable.
            for (let attempt = 0; ; attempt++) {
              const response = await fetch('mms/translate', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: row.source}), signal: pendingTranslation.signal
              });
              const result = await response.json();
              if (version !== translationGeneration) return;
              if (response.status === 429 && attempt < 3) {
                await new Promise(resolve => {
                  const done = () => {
                    clearTimeout(timer);
                    pendingTranslation.signal.removeEventListener('abort', done);
                    resolve();
                  };
                  const timer = setTimeout(done, 1000 * (2 ** attempt));
                  pendingTranslation.signal.addEventListener('abort', done, {once: true});
                });
                if (version !== translationGeneration) return;
                continue;
              }
              if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
              if (!result.text?.trim()) throw new Error('No English text was produced.');
              row.state = 'done';
              row.text = `${result.text}\n  Translation latency: ${result.latency.toFixed(2)} s`;
              if (result.truncated) row.text += '; output reached its length limit and may be incomplete.';
              break;
            }
          } catch (error) {
            if (version !== translationGeneration) return;
            row.state = 'failed'; row.error = error.message;
          }
          renderEnglish();
        }
      } finally {
        if (version === translationGeneration) {
          translationController = null; translationSending = false;
          renderEnglish();
        }
      }
    }
    function enqueueEnglish(chunk, text) {
      if (!english) return;
      const row = {start: chunk.start, end: chunk.end, source: text, state: 'queued'};
      englishRows.push(row); englishRows = englishRows.slice(-120);
      translationQueue.push(row);
      drainEnglish();
      // Bound the backlog without dropping accepted transcripts: stop capture,
      // flush its partial window, then finish both queues (including queued ASR).
      if (recording && translationQueue.length >= maxWaitingTranslations) {
        captureNotice = 'Capture stopped: English translation cannot keep up. Captured audio and queued translations will finish.';
        finishCapture();
      }
      renderEnglish();
    }
    if (translateButton) translateButton.onclick = () => {
      if (translationSending || translationQueue.length) return;
      for (const row of englishRows.filter(row => row.state === 'failed')) {
        row.state = 'queued'; translationQueue.push(row);
      }
      drainEnglish();
    };
    const stamp = value => `${String(Math.floor(value / 60)).padStart(2, '0')}:${(value % 60).toFixed(2).padStart(5, '0')}`;
    function finishCapture(flush = true) {
      recording = false;
      if (processor) { processor.onaudioprocess = null; processor.disconnect(); }
      if (source) source.disconnect();
      if (muted) muted.disconnect();
      if (microphone) microphone.getTracks().forEach(track => track.stop());
      if (context) { context.close(); context = null; }
      // Reserve one extra slot for the final partial window, even when all
      // eight regular ASR queue slots are occupied at Stop/backpressure.
      if (flush && filled) enqueue(pending.slice(0, filled), true);
      filled = 0;
      start.disabled = sending || queue.length > 0;
      stop.disabled = true;
    }
    async function drain() {
      if (sending) return;
      sending = true;
      try {
        while (queue.length) {
          const chunk = queue.shift();
          if (chunk.generation !== generation) continue;
          controller = new AbortController();
          try {
            const response = await fetch('mms/transcribe', {
              method: 'POST', headers: {'Content-Type': 'application/octet-stream',
                'X-Sample-Rate': String(chunk.rate), 'X-Threshold-DB': String(chunk.threshold)},
              body: chunk.samples.buffer, signal: controller.signal
            });
            const result = await response.json();
            if (chunk.generation !== generation) continue;
            if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
            if (result.text) {
              rows.push(`[${stamp(chunk.start)}–${stamp(chunk.end)}] ${result.text}\n  Decode latency: ${result.latency.toFixed(2)} s; audio: ${(chunk.end - chunk.start).toFixed(2)} s`);
              rows = rows.slice(-120);
              transcript.value = rows.join('\n\n');
              transcript.scrollTop = transcript.scrollHeight;
              enqueueEnglish(chunk, result.text);
            }
            status.textContent = result.skipped ? 'Silent/quiet or very short chunk skipped; no inference.' :
              `Decoded in ${result.latency.toFixed(2)} s. ${queue.length} chunks waiting.`;
            if (result.latency > chunk.end - chunk.start)
              status.textContent += ' Processing is slower than capture; Stop lets pending chunks finish.';
            if (captureNotice) status.textContent += ' ' + captureNotice;
          } catch (error) {
            if (chunk.generation === generation && error.name !== 'AbortError') {
              status.textContent = `Transcription error at ${stamp(chunk.start)}: ${error.message}. Stop and restart to retry.`;
              finishCapture(false);
              queue = [];
            }
          }
        }
      } finally {
        sending = false;
        controller = null;
        if (!recording) {
          start.disabled = false;
          if (!status.textContent.includes('error')) status.textContent = 'Stopped. ' + status.textContent;
        }
        renderEnglish();
      }
    }
    function enqueue(samples, finalPartial = false) {
      if (queue.length >= 8 && !finalPartial) {
        finishCapture(false);
        captureNotice = 'Capture stopped: backend cannot keep up. Remaining microphone audio was dropped; queued chunks will finish.';
        status.textContent = captureNotice;
        return;
      }
      const rate = context ? context.sampleRate : pending.rate;
      const begin = received;
      received += samples.length / rate;
      queue.push({samples, rate, threshold: Number(threshold.value), start: begin,
                  end: received, generation});
      drain();
    }
    start.onclick = async () => {
      if (recording || sending || queue.length) return;
      start.disabled = true;
      resetEnglish();
      const currentGeneration = ++generation;
      let newContext, newMicrophone;
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone needs HTTPS or localhost and browser permission.');
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        newContext = new AudioContext();
        await newContext.resume();
        newMicrophone = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1}, video: false});
        if (currentGeneration !== generation) {
          newMicrophone.getTracks().forEach(track => track.stop()); newContext.close(); return;
        }
        context = newContext; microphone = newMicrophone;
        rows = []; transcript.value = ''; received = 0; filled = 0; captureNotice = '';
        pending = new Float32Array(Math.round(seconds * context.sampleRate));
        pending.rate = context.sampleRate;
        source = context.createMediaStreamSource(microphone);
        processor = context.createScriptProcessor(4096, 1, 1);
        muted = context.createGain(); muted.gain.value = 0;
        processor.onaudioprocess = event => {
          if (!recording) return;
          const input = event.inputBuffer.getChannelData(0);
          let offset = 0;
          while (offset < input.length && recording) {
            const count = Math.min(input.length - offset, pending.length - filled);
            pending.set(input.subarray(offset, offset + count), filled);
            filled += count; offset += count;
            if (filled === pending.length) {
              enqueue(pending);
              const rate = pending.rate;
              pending = new Float32Array(pending.length); pending.rate = rate;
              filled = 0;
            }
          }
        };
        source.connect(processor); processor.connect(muted); muted.connect(context.destination);
        recording = true; stop.disabled = false;
        status.textContent = 'Listening. Stop sends the final captured partial chunk.';
        renderEnglish();
      } catch (error) {
        if (newContext && newContext !== context) newContext.close();
        if (currentGeneration !== generation) return;
        finishCapture(false);
        status.textContent = `Microphone error: ${error.message}`;
      }
    };
    stop.onclick = () => { finishCapture(); status.textContent = sending ? 'Stopped. Finishing pending chunks...' : 'Stopped.'; renderEnglish(); };
    clear.onclick = () => {
      resetEnglish();
      ++generation; finishCapture(false); queue = [];
      if (controller) controller.abort();
      rows = []; transcript.value = ''; captureNotice = ''; status.textContent = 'Cleared. Start for a new session.';
      start.disabled = sending;
      renderEnglish();
    };
    window.addEventListener('pagehide', () => { resetEnglish(); ++generation; finishCapture(false); if (controller) controller.abort(); });
  };
  bind();
}"""


def microphone_routes(engine):
    """Receive bounded PCM chunks in memory; no Gradio upload/cache path."""
    from starlette.concurrency import run_in_threadpool
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def transcribe(request):
        try:
            rate = int(request.headers.get("X-Sample-Rate", "0"))
            threshold = float(request.headers.get("X-Threshold-DB", "-45"))
            if not 8000 <= rate <= 192000 or not -60 <= threshold <= -20:
                raise ValueError("Invalid sample rate or silence threshold.")
            body = bytearray()
            async for part in request.stream():
                body.extend(part)
                if len(body) > rate * 5 * 4:
                    raise ValueError("Audio chunks must be no longer than five seconds.")
            if len(body) % 4:
                raise ValueError("Expected float32 PCM bytes.")
            samples = prepare_audio((rate, np.frombuffer(body, dtype="<f4")))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        try:
            result = await run_in_threadpool(engine.transcribe, samples, threshold)
            return JSONResponse({"text": result.text, "latency": result.latency, "skipped": result.skipped})
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return [Route("/mms/transcribe", transcribe, methods=["POST"])]


TRANSCRIPT_CSS = """
/* Pair native control colors explicitly; Gradio's dark text otherwise
   inherits onto the browser's default white textarea background. */
#mms-transcript, #mms-english, #mms-threshold {
  --mms-foreground: #111827;
  --mms-background: #ffffff;
  --mms-placeholder: #4b5563;
  --mms-border: #6b7280;
  color-scheme: light;
}
.dark #mms-transcript {
  --mms-foreground: #f3f4f6;
  --mms-background: #111827;
  --mms-placeholder: #cbd5e1;
  --mms-border: #9ca3af;
  color-scheme: dark;
}
.dark #mms-english {
  --mms-foreground: #f3f4f6;
  --mms-background: #111827;
  --mms-placeholder: #cbd5e1;
  --mms-border: #9ca3af;
  color-scheme: dark;
}
.dark #mms-threshold {
  --mms-foreground: #f3f4f6;
  --mms-background: #111827;
  --mms-placeholder: #cbd5e1;
  --mms-border: #9ca3af;
  color-scheme: dark;
}
#mms-transcript, #mms-transcript:read-only, #mms-transcript:disabled,
#mms-english, #mms-english:read-only, #mms-english:disabled,
#mms-threshold, #mms-threshold:disabled {
  color: var(--mms-foreground);
  -webkit-text-fill-color: var(--mms-foreground);
  background-color: var(--mms-background);
  border: 1px solid var(--mms-border);
  opacity: 1;
}
#mms-transcript::placeholder, #mms-english::placeholder, #mms-threshold::placeholder {
  color: var(--mms-placeholder);
  -webkit-text-fill-color: var(--mms-placeholder);
  opacity: 1;
}
"""


def build_demo(engine, chunk_seconds=3.0, *, english_translation=False):
    import gradio as gr

    with gr.Blocks(title="Hakha Chin · Meta MMS", js=RECORDER_JS, css=TRANSCRIPT_CSS) as demo:
        gr.Markdown(f"# Hakha Chin speech-to-text\n"
                    f"Meta MMS · `cnh` · {engine.device.upper()} · {chunk_seconds:g}-second chunks\n\n"
                    "Use **Start microphone** and **Stop**. Text arrives after each chunk "
                    "is processed. Timestamp ranges describe received audio; decode latency excludes "
                    "capture, transport, and queue time. This is intermittent transcription. "
                    "Accuracy on conversational Hakha Chin has not been evaluated.")
        gr.Markdown("**License: Meta MMS weights — [CC BY-NC 4.0 (noncommercial)]"
                    "(https://creativecommons.org/licenses/by-nc/4.0/).** "
                    "Credit: [Meta / facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all), "
                    "Pratap et al., 2023. Commercial use requires separate license consideration.\n\n"
                    "**Audio privacy:** Your microphone audio is sent to this Python backend. "
                    "Inference runs here; audio is not sent to a hosted ASR service. "
                    "Audio chunks stay in browser/backend memory; this demo does not save recordings to disk. "
                    "Clear discards waiting browser chunks and ignores results from an in-progress request. "
                    "A share URL routes traffic through Gradio's public tunnel. Record only with consent.")
        gr.HTML(f"""<div style="display:flex;gap:12px;flex-wrap:wrap;margin:16px 0">
          <button id="mms-start" data-seconds="{chunk_seconds}" data-english="{str(english_translation).lower()}" style="padding:14px">Start microphone</button>
          <button id="mms-stop" disabled style="padding:14px">Stop</button>
          <button id="mms-clear" style="padding:14px">Clear / reset</button></div>
          <label>Silence gate (dBFS): <input id="mms-threshold" type="number" min="-60" max="-20" value="-45" /></label>
          <p>Lower values accept quieter audio and more noise. This energy gate cannot identify speech.</p>
          <p id="mms-status" role="status" aria-live="polite">Ready. Allow microphone access, then Start.</p>
          <label for="mms-transcript">Hakha Chin transcript (last 120 chunks)</label>
          <textarea id="mms-transcript" readonly rows="12" style="width:100%;padding:12px" aria-label="Hakha Chin transcript" placeholder="Transcribed Hakha Chin text will appear here."></textarea>""")
        if english_translation:
            gr.HTML("""<section aria-label="Experimental English translation">
              <h2>English attempt — unverified</h2>
              <p>Each transcription chunk is sent for English translation as it arrives, while the
              microphone keeps recording. English appears below in timestamp order. Transcription
              mistakes can produce misleading English; verify with a Hakha Chin speaker.</p>
              <label for="mms-english">Attempted English translation (last 120 chunks)</label>
              <textarea id="mms-english" readonly rows="6" style="width:100%;padding:12px"
                aria-label="Attempted English translation" placeholder="English translation will appear here as transcription chunks arrive."></textarea>
              <p id="mms-english-status" role="status" aria-live="polite">Ready for live English translation.</p>
              <button id="mms-translate" disabled style="padding:12px">Retry English</button>
              <p>Stop finishes captured audio and queued translations. If eight translations are waiting,
              capture stops automatically so processing can catch up. First use loads the translation model.</p>
              <p>MADLAD-400 3B runs on this same CPU server. No transcript is sent to a separate translation service.
              Model: <a href="https://huggingface.co/google/madlad400-3b-mt">Google MADLAD-400</a>, Apache 2.0;
              <a href="https://huggingface.co/Heng666/madlad400-3b-mt-ct2-int8">INT8 conversion by Heng-Shiou Sheu</a>.</p>
            </section>""")
    return demo


def smoke_test(engine, audio_path=None, chunk_seconds=3.0):
    """Real model load + skipped input + actual forward pass; no accuracy claim."""
    checks = {
        "model": MODEL_ID, "revision": engine.model.config._commit_hash,
        "adapter": engine.model.target_lang,
        "tokenizer_language": engine.processor.tokenizer.target_lang,
        "device": engine.device,
    }
    for name, samples in (("empty", np.empty(0, np.float32)),
                          ("silence", np.zeros(SAMPLE_RATE * 3, np.float32))):
        result = engine.transcribe(samples)
        assert result.skipped and result.text == "" and result.latency == 0
        checks[name] = "skipped, empty text, no inference"
    # Deterministic non-speech signal exercises real CTC inference, even when no
    # consented Hakha Chin recording is available. It is not a recognition test.
    timeline = np.arange(SAMPLE_RATE * 2, dtype=np.float32) / SAMPLE_RATE
    tone = (0.1 * np.sin(2 * np.pi * 440 * timeline)).astype(np.float32)
    result = engine.transcribe(tone)
    assert not result.skipped
    checks["synthetic_tone"] = {"text_verbatim": result.text, "decode_seconds": result.latency,
                                "note": "Non-speech forward-pass check; no accuracy evidence."}
    if audio_path:
        import soundfile as sf

        samples, rate = sf.read(audio_path, dtype="float32", always_2d=False)
        samples = prepare_audio((rate, samples))
        session = Session()
        text, status = consume(session, samples, engine, chunk_seconds, -45, flush=True)
        checks["audio"] = {"file": str(audio_path), "transcript_verbatim": text, "status": status}
        if "Error at" in status:
            raise RuntimeError(status)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--chunk-seconds", type=float, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Opt in to a public Gradio HTTPS tunnel.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--audio", help="Consented audio file to transcribe during --smoke-test.")
    args = parser.parse_args()
    if not 2 <= args.chunk_seconds <= 5:
        parser.error("--chunk-seconds must be between 2 and 5.")
    if args.audio and not args.smoke_test:
        parser.error("--audio requires --smoke-test.")
    try:
        engine = MMSEngine(args.device)
    except Exception as exc:
        raise SystemExit(f"MMS initialization failed: {type(exc).__name__}: {exc}\n"
                         "Check requirements-mms.txt, Hugging Face connectivity/cache, free RAM, "
                         "and the requested device. No alternate model is used.") from exc
    if args.smoke_test:
        smoke_test(engine, args.audio, args.chunk_seconds)
    else:
        build_demo(engine, args.chunk_seconds).launch(
            server_name=args.host, server_port=args.port, share=args.share, show_error=True,
            app_kwargs={"routes": microphone_routes(engine)})


if __name__ == "__main__":
    main()
