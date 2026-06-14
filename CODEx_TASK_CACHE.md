# Codex task: add LRU caching for repeated translations and TTS output

Repository: `trinitron88/ChinTranslator`
Base branch: `main`
Issue: #10

## Goal
Reduce realtime latency, network calls, and phone battery drain in `hf_space/app.py` by caching repeated translation and TTS outputs.

## Implementation notes
- Work primarily in `hf_space/app.py`.
- Add normalized-text cache keys for translation:
  - key: `(normalized_text, source_lang, target_lang)`
  - normalize by trimming and collapsing whitespace.
- Use `functools.lru_cache(maxsize=512)` or a small explicit LRU.
- Add a separate TTS cache if low-risk:
  - key: `(normalized_text, tts_lang)`
  - value: `(sample_rate, int16_pcm)` or an immutable/copy-safe representation.
- Do not return a mutable cached NumPy array directly without copying it.
- Keep behavior identical on cache misses.
- Empty or whitespace-only input must still return immediately.
- Add lightweight cache-hit/miss logging only if it is not too noisy, preferably env-gated.

## Acceptance criteria
- Repeated identical utterances avoid duplicate Google Translate requests.
- Repeated identical TTS phrases avoid regenerating MP3 output.
- App still works in Hugging Face Spaces without persistent filesystem.
- No change to the external UI is required.

## Suggested validation
- Run or import `hf_space/app.py` enough to confirm syntax/imports.
- Manually call translation twice with the same text and confirm the second call uses cache or cache stats increment.
- Manually call TTS twice with the same text if TTS cache is implemented.
