const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const path = require('path');

// Exercise the actual recorder through the browser interfaces it uses. Deferred
// fetches let these tests reproduce slow translation and stale request races.
const pythonSource = fs.readFileSync(path.join(__dirname, '..', 'mms_realtime.py'), 'utf8');
const source = pythonSource.match(/RECORDER_JS = r"""([\s\S]*?)"""/)[1];
const tick = () => new Promise(resolve => setImmediate(resolve));
async function settle() {
  await tick();
  await tick();
}

function harness({ english = true, asrResults = [], deferredASR = false } = {}) {
  const ids = ['mms-start', 'mms-stop', 'mms-clear', 'mms-status', 'mms-transcript', 'mms-threshold'];
  if (english) ids.push('mms-english', 'mms-english-status', 'mms-translate');
  const el = Object.fromEntries(ids.map(id => [id, {
    dataset: {}, value: '', textContent: '', disabled: false, scrollTop: 0, scrollHeight: 0,
  }]));
  el['mms-start'].dataset = { seconds: '3', english: String(english) };
  el['mms-stop'].disabled = true;
  el['mms-threshold'].value = '-45';
  const asrRequests = [], translationRequests = [], events = {}, timers = new Map();
  let processor, trackStops = 0, timerId = 0;
  const audioNode = () => ({ connect() {}, disconnect() {}, gain: { value: 1 } });
  class AudioContext {
    constructor() { this.sampleRate = 16000; this.destination = {}; }
    resume() {}
    close() {}
    createMediaStreamSource() { return audioNode(); }
    createScriptProcessor() { processor = audioNode(); return processor; }
    createGain() { return audioNode(); }
  }
  const response = (result, status = 200) => ({
    ok: status >= 200 && status < 300,
    status,
    headers: { get: () => null },
    json: async () => result,
  });
  const deferred = (list, options) => new Promise((resolve, reject) => {
    list.push({ options, resolve, reject, settled: false });
  });
  const context = {
    document: { getElementById: id => el[id] || null },
    window: { AudioContext, addEventListener: (id, cb) => { events[id] = cb; } },
    navigator: { mediaDevices: { getUserMedia: async () => ({
      getTracks: () => [{ stop() { trackStops += 1; } }],
    }) } },
    requestAnimationFrame: cb => cb(),
    AbortController, Float32Array, console,
    setTimeout: (callback, delay) => {
      const id = ++timerId;
      timers.set(id, { callback, delay });
      return id;
    },
    clearTimeout: id => timers.delete(id),
    fetch: (url, options) => {
      if (url === 'mms/transcribe') {
        if (deferredASR) return deferred(asrRequests, options);
        const index = asrRequests.length;
        const result = { text: `Chin ${index + 1}`, latency: 0.5, skipped: false, ...asrResults[index] };
        asrRequests.push({ options, result });
        return Promise.resolve(response(result));
      }
      assert.equal(url, 'mms/translate', 'recorder should use the local translation endpoint');
      return deferred(translationRequests, options);
    },
  };
  vm.runInNewContext('(' + source + ')()', context);
  function resolveRequest(list, index, result, status) {
    const request = list[index];
    assert(request, `request ${index} must exist before resolving it`);
    assert(!request.settled, `request ${index} must only be resolved once`);
    request.settled = true;
    request.resolve(response(result, status));
  }
  return {
    el, asrRequests, translationRequests, events, timers,
    get trackStops() { return trackStops; },
    get recording() { return !el['mms-stop'].disabled; },
    get transcript() { return el['mms-transcript'].value; },
    get englishText() { return el['mms-english'].value; },
    start: () => el['mms-start'].onclick(),
    stop: () => el['mms-stop'].onclick(),
    clear: () => el['mms-clear'].onclick(),
    retry: () => el['mms-translate'].onclick(),
    async chunk(sampleCount = 48000) {
      assert(processor?.onaudioprocess, 'capture must be active to deliver microphone audio');
      processor.onaudioprocess({ inputBuffer: { getChannelData: () => new Float32Array(sampleCount) } });
      await settle();
    },
    translatedSource(index) { return JSON.parse(translationRequests[index].options.body).text; },
    resolveTranslation(index, text = `English ${index + 1}`, extra = {}) {
      resolveRequest(translationRequests, index, { text, latency: 1, truncated: false, ...extra }, 200);
    },
    failTranslation(index, status = 500, error = 'translation unavailable') {
      resolveRequest(translationRequests, index, { error }, status);
    },
    resolveASR(index, text, extra = {}) {
      resolveRequest(asrRequests, index, { text, latency: 0.5, skipped: false, ...extra }, 200);
    },
    async nextTimer() {
      const timer = timers.entries().next().value;
      assert(timer, 'a retry timer should be scheduled');
      timers.delete(timer[0]);
      timer[1].callback();
      await settle();
    },
  };
}

function ranges(text) {
  return text.match(/^\[\d+:\d{2}\.\d{2}[–-]\d+:\d{2}\.\d{2}\]/gm) || [];
}
function englishRows(h) {
  return h.englishText.split(/(?=^\[\d+:\d{2}\.\d{2}[–-])/m).filter(row => row.trim());
}
const tests = [];
function test(name, run) { tests.push({ name, run }); }

test('streams exact chunks before Stop and keeps ASR independent of a slow translator', async () => {
  const h = harness();
  await h.start();
  await h.chunk();
  assert(h.recording);
  assert.equal(h.translationRequests.length, 1, 'first transcript must translate while recording');
  assert.equal(h.translatedSource(0), 'Chin 1');
  for (let i = 0; i < 4; i++) await h.chunk();
  assert.equal(h.asrRequests.length, 5, 'ASR must continue while the first translation is stalled');
  assert.equal(h.translationRequests.length, 1, 'translation requests must be serial');
  assert.equal(ranges(h.englishText).length, 5, 'all accepted chunks must have timestamped English rows');
  assert.match(h.englishText, /pending|waiting|translat/i, 'pending English should be visible');
  const transcript = h.transcript;
  for (let i = 0; i < 5; i++) {
    assert.equal(h.translationRequests.length, i + 1);
    assert.equal(h.translatedSource(i), `Chin ${i + 1}`, 'translate the chunk, not a combined excerpt');
    h.resolveTranslation(i);
    await settle();
  }
  const rows = englishRows(h);
  assert.equal(rows.length, 5);
  for (let i = 0; i < rows.length; i++) assert(rows[i].includes(`English ${i + 1}`));
  assert.deepEqual(ranges(h.englishText), ranges(h.transcript), 'English timestamps must match transcript order');
  assert.equal(h.transcript, transcript, 'English output must not replace the original transcript');
  h.stop();
  await settle();
  assert.equal(h.translationRequests.length, 5, 'Stop must not translate completed chunks again');
});

test('skips empty ASR and translates repeated text as separate chunks', async () => {
  const h = harness({ asrResults: [
    { text: '', skipped: true }, { text: '', skipped: false },
    { text: 'Same phrase' }, { text: 'Same phrase' },
  ] });
  await h.start();
  await h.chunk();
  await h.chunk();
  assert.equal(h.translationRequests.length, 0);
  await h.chunk();
  await h.chunk();
  h.resolveTranslation(0, 'Same English');
  await settle();
  assert.equal(h.translationRequests.length, 2, 'identical text at different timestamps is separate speech');
  assert.equal(h.translatedSource(0), 'Same phrase');
  assert.equal(h.translatedSource(1), 'Same phrase');
  h.resolveTranslation(1, 'Same English');
  await settle();
  assert.equal((h.englishText.match(/Same English/g) || []).length, 2);
  assert.deepEqual(ranges(h.englishText), ranges(h.transcript));
  h.stop();
});

test('Stop flushes the partial chunk and drains both workers without duplication', async () => {
  const h = harness({ deferredASR: true });
  await h.start();
  await h.chunk();
  await h.chunk(16000);
  assert.equal(h.asrRequests.length, 1);
  h.stop();
  assert(!h.recording);
  h.resolveASR(0, 'Full chunk');
  await settle();
  assert.equal(h.asrRequests.length, 2, 'Stop must send the captured partial chunk');
  assert.equal(h.asrRequests[0].options.body.byteLength, 48000 * 4);
  assert.equal(h.asrRequests[1].options.body.byteLength, 16000 * 4);
  assert.equal(h.translatedSource(0), 'Full chunk', 'finished ASR should translate while partial ASR runs');
  h.resolveASR(1, 'Partial chunk');
  await settle();
  assert.equal(h.translationRequests.length, 1);
  h.resolveTranslation(0, 'Full English');
  await settle();
  assert.equal(h.translatedSource(1), 'Partial chunk');
  h.resolveTranslation(1, 'Partial English');
  await settle();
  assert.equal(ranges(h.englishText).length, 2);
  assert.deepEqual(ranges(h.englishText), ['[00:00.00–00:03.00]', '[00:03.00–00:04.00]']);
  assert.equal(h.el['mms-start'].disabled, false);
  h.stop();
  await settle();
  assert.equal(h.asrRequests.length, 2);
  assert.equal(h.translationRequests.length, 2);
});

test('Clear cancels active translation, discards queued jobs, and ignores a late response', async () => {
  const h = harness();
  await h.start();
  for (let i = 0; i < 3; i++) await h.chunk();
  h.clear();
  assert(h.translationRequests[0].options.signal.aborted);
  assert.equal(h.englishText, '');
  assert.equal(h.transcript, '');
  h.resolveTranslation(0, 'Stale translation');
  await settle();
  assert.equal(h.englishText, '');
  assert.equal(h.translationRequests.length, 1, 'Clear must drop pending translation jobs');
  await h.start();
  await h.chunk();
  assert.equal(h.translatedSource(1), 'Chin 4');
  h.resolveTranslation(1, 'New session');
  await settle();
  assert.match(h.englishText, /New session/);
  assert.doesNotMatch(h.englishText, /Stale translation/);
  h.stop();
});

test('new Start replaces active translation and queued jobs from the previous recording', async () => {
  const h = harness();
  await h.start();
  await h.chunk();
  await h.chunk();
  h.stop();
  await h.start();
  assert(h.translationRequests[0].options.signal.aborted);
  assert.equal(h.englishText, '');
  await h.chunk();
  h.resolveTranslation(0, 'Previous recording');
  await settle();
  assert.equal(h.translationRequests.length, 2);
  assert.equal(h.translatedSource(1), 'Chin 3', 'the old queued chunk must not be sent');
  h.resolveTranslation(1, 'Current recording');
  await settle();
  assert.doesNotMatch(h.englishText, /Previous recording/);
  assert.match(h.englishText, /Current recording/);
  assert.equal(ranges(h.englishText).length, 1);
  h.stop();
});

test('a failed translation does not block later chunks and Retry only resends failed rows', async () => {
  const h = harness();
  await h.start();
  await h.chunk();
  await h.chunk();
  const transcript = h.transcript;
  h.failTranslation(0);
  await settle();
  assert.equal(h.translationRequests.length, 2, 'a failed job must release the serial worker');
  assert.match(englishRows(h)[0], /fail|error/i);
  h.resolveTranslation(1, 'Successful second');
  await settle();
  h.stop();
  assert.equal(h.transcript, transcript);
  assert.equal(h.el['mms-translate'].disabled, false);
  h.retry();
  await settle();
  assert.equal(h.translationRequests.length, 3);
  assert.equal(h.translatedSource(2), 'Chin 1');
  h.resolveTranslation(2, 'Recovered first');
  await settle();
  assert.equal(ranges(h.englishText).length, 2, 'retry must update the failed row in place');
  const rows = englishRows(h);
  assert.match(rows[0], /Recovered first/);
  assert.match(rows[1], /Successful second/);
  assert.equal((h.englishText.match(/Successful second/g) || []).length, 1);
  h.retry();
  await settle();
  assert.equal(h.translationRequests.length, 3, 'successful rows must never be retried');
});

test('truncation remains visible on its row after later chunks complete', async () => {
  const h = harness();
  await h.start();
  await h.chunk();
  await h.chunk();
  h.resolveTranslation(0, 'First English', { truncated: true });
  await settle();
  h.resolveTranslation(1, 'Complete English');
  await settle();
  const rows = englishRows(h);
  assert.match(rows[0], /truncat|incomplete|length limit/i);
  assert.match(rows[1], /Complete English/);
  h.stop();
});

test('empty English remains retryable and does not block the next chunk', async () => {
  const h = harness();
  await h.start();
  await h.chunk();
  await h.chunk();
  h.resolveTranslation(0, '   ');
  await settle();
  assert.match(englishRows(h)[0], /fail|error/i);
  assert.equal(h.translatedSource(1), 'Chin 2');
  h.resolveTranslation(1, 'Second English');
  await settle();
  assert.equal(h.el['mms-translate'].disabled, false);
  h.stop();
});

test('busy responses have bounded retries and eventually allow the next chunk', async () => {
  const h = harness();
  await h.start();
  await h.chunk();
  await h.chunk();
  let rejected = 0;
  while (h.translationRequests.length > rejected && h.translatedSource(rejected) === 'Chin 1') {
    assert(rejected < 6, 'HTTP 429 retries must be bounded');
    h.failTranslation(rejected++, 429, 'translator busy');
    await settle();
    if (h.timers.size) await h.nextTimer();
  }
  assert(rejected >= 1);
  assert.equal(h.translatedSource(rejected), 'Chin 2', 'a persistently busy row must not block the next chunk');
  h.resolveTranslation(rejected, 'Second after busy');
  await settle();
  assert.match(englishRows(h)[0], /fail|error/i);
  assert.match(englishRows(h)[1], /Second after busy/);
  h.stop();
});

test('Clear during retry backoff cancels the timer and all pending translations', async () => {
  const h = harness();
  await h.start();
  await h.chunk();
  await h.chunk();
  h.failTranslation(0, 429, 'translator busy');
  await settle();
  assert.equal(h.timers.size, 1);
  h.clear();
  await settle();
  assert(h.translationRequests[0].options.signal.aborted);
  assert.equal(h.timers.size, 0, 'Clear must cancel scheduled retries');
  assert.equal(h.translationRequests.length, 1);
  assert.equal(h.englishText, '');
  await h.start();
  await h.chunk();
  assert.equal(h.translatedSource(1), 'Chin 3');
  h.resolveTranslation(1, 'New English');
  await settle();
  assert.equal(ranges(h.englishText).length, 1);
  h.stop();
});

test('Clear ignores late ASR and does not send its text to the translator', async () => {
  const h = harness({ deferredASR: true });
  await h.start();
  await h.chunk();
  await h.chunk();
  h.clear();
  assert(h.asrRequests[0].options.signal.aborted);
  h.resolveASR(0, 'Stale transcription');
  await settle();
  assert.equal(h.asrRequests.length, 1, 'Clear must discard queued audio');
  assert.equal(h.translationRequests.length, 0);
  assert.equal(h.transcript, '');
  assert.equal(h.englishText, '');
  assert.equal(h.el['mms-start'].disabled, false);
});

test('a bounded translation backlog stops capture and finishes all accepted chunks', async () => {
  const h = harness();
  await h.start();
  let captured = 0;
  while (h.recording && captured < 20) {
    await h.chunk();
    captured += 1;
  }
  assert(!h.recording, 'a stalled translator must not allow an unbounded backlog');
  assert(captured >= 8, 'capture should tolerate a small backlog before stopping');
  assert(h.trackStops > 0, 'automatic stop must release the microphone');
  assert.equal(h.translationRequests.length, 1);
  const accepted = ranges(h.transcript).length;
  assert.equal(accepted, h.asrRequests.length);
  for (let i = 0; i < accepted; i++) {
    assert.equal(h.translatedSource(i), `Chin ${i + 1}`);
    h.resolveTranslation(i, `Backlog English ${i + 1}`);
    await settle();
  }
  assert.equal(h.translationRequests.length, accepted);
  assert.deepEqual(ranges(h.englishText), ranges(h.transcript));
  assert.equal((h.englishText.match(/Backlog English/g) || []).length, accepted);
  assert.equal(h.el['mms-start'].disabled, false);
});

test('manual Stop preserves partial audio even when the ASR queue is full', async () => {
  const h = harness({ deferredASR: true });
  await h.start();
  // ASR is stalled: one request, eight queued full chunks, and a partial
  // microphone window. Stop must reserve space for that final partial window.
  for (let i = 0; i < 9; i++) await h.chunk();
  await h.chunk(16000);
  assert(h.recording);
  h.stop();
  for (let i = 0; i < 9; i++) {
    h.resolveASR(i, `Captured ${i + 1}`);
    await settle();
  }
  assert(!h.recording);
  assert.equal(h.asrRequests.length, 10, 'Stop must flush the partial window even at the queue limit');
  assert.equal(h.asrRequests[9].options.body.byteLength, 16000 * 4);
  h.resolveASR(9, 'Final partial');
  await settle();
  for (let i = 0; i < 10; i++) {
    assert.equal(h.translatedSource(i), i === 9 ? 'Final partial' : `Captured ${i + 1}`);
    h.resolveTranslation(i, `Translated ${i + 1}`);
    await settle();
  }
  assert.deepEqual(ranges(h.englishText), ranges(h.transcript));
  assert.equal(ranges(h.englishText).at(-1), '[00:27.00–00:28.00]');
  assert.equal(h.el['mms-start'].disabled, false);
});

test('translation backlog auto-stop preserves the partial window alongside a full ASR queue', async () => {
  const h = harness({ deferredASR: true });
  await h.start();
  // Hold English at one active request and seven waiting chunks.
  for (let i = 0; i < 8; i++) {
    await h.chunk();
    h.resolveASR(i, `Captured ${i + 1}`);
    await settle();
  }
  // Hold the next ASR request, fill all eight ASR waiting slots, then capture
  // a partial window. Resolving ASR now triggers translation's automatic Stop.
  for (let i = 0; i < 9; i++) await h.chunk();
  await h.chunk(16000);
  assert(h.recording);
  assert.equal(h.asrRequests.length, 9);
  h.resolveASR(8, 'Captured 9');
  await settle();
  assert(!h.recording);
  for (let i = 9; i < 17; i++) {
    h.resolveASR(i, `Captured ${i + 1}`);
    await settle();
  }
  assert.equal(h.asrRequests.length, 18, 'automatic Stop must preserve the partial window at the ASR queue limit');
  assert.equal(h.asrRequests[17].options.body.byteLength, 16000 * 4);
  h.resolveASR(17, 'Final partial');
  await settle();
  for (let i = 0; i < 18; i++) {
    assert.equal(h.translatedSource(i), i === 17 ? 'Final partial' : `Captured ${i + 1}`);
    h.resolveTranslation(i, `Translated ${i + 1}`);
    await settle();
  }
  assert.deepEqual(ranges(h.englishText), ranges(h.transcript));
  assert.equal(ranges(h.englishText).at(-1), '[00:51.00–00:52.00]');
  assert.equal(h.el['mms-start'].disabled, false);
});

test('the transcription-only page works without any English controls', async () => {
  const h = harness({ english: false });
  await h.start();
  await h.chunk();
  await h.chunk(16000);
  h.stop();
  await settle();
  assert.equal(h.asrRequests.length, 2);
  assert.equal(h.translationRequests.length, 0);
  assert.equal(ranges(h.transcript).length, 2);
  h.clear();
  assert.equal(h.transcript, '');
});

(async () => {
  let failures = 0;
  for (const { name, run } of tests) {
    try {
      await run();
      console.log(`PASS ${name}`);
    } catch (error) {
      failures += 1;
      console.error(`FAIL ${name}`);
      console.error(error);
    }
  }
  if (failures) process.exitCode = 1;
  else console.log(`Recorder translation tests passed (${tests.length} scenarios).`);
})().catch(error => { console.error(error); process.exitCode = 1; });
