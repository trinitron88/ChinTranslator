"""Offline regression tests; no model weights or Hugging Face access needed."""

import unittest

import httpx
import numpy as np
from starlette.applications import Starlette

import mms_realtime as mms


def tone(seconds=1.0, amplitude=0.1, sample_rate=mms.SAMPLE_RATE):
    timeline = np.arange(round(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 240 * timeline)).astype(np.float32)


class FakeEngine:
    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.calls = []

    def transcribe(self, samples, threshold_db):
        self.calls.append((samples.copy(), threshold_db))
        index = len(self.calls) - 1
        result = self.outcomes[index] if index < len(self.outcomes) else mms.Result(
            text=f"Chunk {index + 1}", latency=0.1
        )
        if isinstance(result, Exception):
            raise result
        return result


class PrepareAudioTests(unittest.TestCase):
    def test_signed_pcm_has_correct_scale_and_dtype(self):
        result = mms.prepare_audio((16000, np.array([-32768, 0, 16384, 32767], np.int16)))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, [-1.0, 0.0, 0.5, 32767 / 32768])

    def test_unsigned_pcm_is_centered_at_midpoint(self):
        result = mms.prepare_audio((16000, np.array([0, 128, 192, 255], np.uint8)))
        np.testing.assert_array_equal(result, [-1.0, 0.0, 0.5, 127 / 128])

    def test_float_stereo_is_mixed_to_mono_and_clipped(self):
        samples = np.array([[0.6, 0.2], [-0.6, 0.6], [3, 1], [-1, -3]], np.float64)
        result = mms.prepare_audio((16000, samples))
        self.assertEqual(result.shape, (4,))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, [0.4, 0, 1, -1], atol=1e-7)

    def test_resampling_preserves_duration_and_signal(self):
        for sample_rate in (8000, 44100, 48000):
            with self.subTest(sample_rate=sample_rate):
                result = mms.prepare_audio((sample_rate, tone(sample_rate=sample_rate)))
                self.assertEqual(result.shape, (16000,))
                self.assertEqual(result.dtype, np.float32)
                # Ignore filter transients at the recording edges. The same
                # physical tone should retain its frequency and amplitude.
                np.testing.assert_allclose(result[160:-160], tone()[160:-160], atol=0.001)

    def test_empty_input_remains_empty(self):
        for audio in (None, (16000, np.empty(0)), (48000, np.empty((0, 2), np.int16))):
            with self.subTest(audio=audio):
                result = mms.prepare_audio(audio)
                self.assertEqual(result.shape, (0,))
                self.assertEqual(result.dtype, np.float32)

    def test_invalid_sample_rates_are_rejected(self):
        for sample_rate in (0, -16000, 16000.0, "16000", None):
            with self.subTest(sample_rate=sample_rate):
                with self.assertRaisesRegex(ValueError, "sample rate"):
                    mms.prepare_audio((sample_rate, np.ones(10)))

    def test_invalid_shapes_and_sample_types_are_rejected(self):
        for samples in (np.array(1), np.ones((2, 2, 2)), np.ones((10, 9)),
                        np.array(["audio"]), np.array([1j]), np.array([True])):
            with self.subTest(shape=samples.shape, dtype=samples.dtype):
                with self.assertRaises(ValueError):
                    mms.prepare_audio((16000, samples))

    def test_nonfinite_audio_is_rejected_before_mixing_or_clipping(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                    mms.prepare_audio((16000, np.array([[value, 0], [0.1, -0.1]])))


class SilenceGateTests(unittest.TestCase):
    def test_empty_silence_quiet_audio_dc_and_short_audio_are_skipped(self):
        inputs = {
            "empty": np.empty(0, np.float32),
            "silence": np.zeros(16000, np.float32),
            "quiet": tone(amplitude=0.0001),
            "dc_offset": np.full(16000, 0.7, np.float32),
            "short": tone(seconds=0.24),
        }
        # Bypassing __init__ deliberately leaves model, processor, torch, and
        # lock absent: a skipped input must return before touching inference.
        engine = mms.MMSEngine.__new__(mms.MMSEngine)
        for label, samples in inputs.items():
            with self.subTest(label=label):
                self.assertFalse(mms.has_signal(samples))
                result = engine.transcribe(samples)
                self.assertTrue(result.skipped)
                self.assertEqual(result.text, "")
                self.assertEqual(result.latency, 0)

    def test_audible_tone_passes_and_threshold_is_respected(self):
        samples = tone(seconds=0.25, amplitude=0.02)
        self.assertTrue(mms.has_signal(samples, threshold_db=-45))
        self.assertFalse(mms.has_signal(samples, threshold_db=-20))

    def test_brief_noise_burst_does_not_open_gate(self):
        brief = np.concatenate((tone(seconds=0.08), np.zeros(16000, np.float32)))
        sustained = np.concatenate((tone(seconds=0.12), np.zeros(16000, np.float32)))
        self.assertFalse(mms.has_signal(brief))
        self.assertTrue(mms.has_signal(sustained))


class ConsumeTests(unittest.TestCase):
    def consume(self, session, samples, engine, flush=False):
        return mms.consume(session, samples, engine, chunk_seconds=1,
                           threshold_db=-42, flush=flush)

    def test_fragments_form_fixed_windows_in_order_with_correct_offsets(self):
        session = mms.Session()
        engine = FakeEngine()
        received = np.arange(36000, dtype=np.float32)

        text, status = self.consume(session, received[:6000], engine)
        self.assertEqual(text, "")
        self.assertIn("collecting", status)
        self.assertEqual(engine.calls, [])
        self.assertEqual(session.seconds, 0)

        text, status = self.consume(session, received[6000:], engine)
        self.assertEqual(len(engine.calls), 2)
        np.testing.assert_array_equal(engine.calls[0][0], received[:16000])
        np.testing.assert_array_equal(engine.calls[1][0], received[16000:32000])
        self.assertEqual([call[1] for call in engine.calls], [-42, -42])
        np.testing.assert_array_equal(session.pending, received[32000:])
        self.assertEqual(session.seconds, 2)
        self.assertIn("[00:00.00–00:01.00] Chunk 1", text)
        self.assertIn("[00:01.00–00:02.00] Chunk 2", text)
        self.assertLess(text.index("Chunk 1"), text.index("Chunk 2"))
        self.assertIn("Decode latency: 0.10 s; audio: 1.00 s", text)

    def test_silent_chunks_advance_time_without_inventing_transcript_rows(self):
        session = mms.Session()
        engine = FakeEngine([mms.Result(skipped=True), mms.Result(text="Hakha Chin", latency=0.2)])
        text, status = self.consume(session, np.zeros(32000, np.float32), engine)
        self.assertEqual(len(session.rows), 1)
        self.assertTrue(text.startswith("[00:01.00–00:02.00] Hakha Chin"))
        self.assertIn("skipped", status)
        self.assertNotIn("skipped", text)
        self.assertEqual(session.seconds, 2)

    def test_stop_flushes_final_partial_chunk_exactly_once(self):
        session = mms.Session()
        engine = FakeEngine()
        samples = np.arange(20000, dtype=np.float32)
        self.consume(session, samples, engine)
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(len(session.pending), 4000)

        text, status = self.consume(session, np.empty(0, np.float32), engine, flush=True)
        self.assertEqual(len(engine.calls), 2)
        np.testing.assert_array_equal(engine.calls[1][0], samples[16000:])
        self.assertIn("[00:01.00–00:01.25] Chunk 2", text)
        self.assertIn("audio: 0.25 s", text)
        self.assertEqual(session.seconds, 1.25)
        self.assertEqual(len(session.pending), 0)

        again, _ = self.consume(session, np.empty(0, np.float32), engine, flush=True)
        self.assertEqual(again, text)
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(session.seconds, 1.25)

    def test_inference_errors_are_consumed_once_and_kept_out_of_transcript(self):
        session = mms.Session()
        engine = FakeEngine([RuntimeError("decoder failed"), mms.Result(text="Recovered", latency=0.1)])
        samples = np.arange(32000, dtype=np.float32)
        text, status = self.consume(session, samples, engine)
        self.assertIn("Error at 00:00.00: RuntimeError: decoder failed", status)
        self.assertNotIn("decoder failed", text)
        self.assertNotIn("Error", text)
        self.assertTrue(text.startswith("[00:01.00–00:02.00] Recovered"))
        self.assertEqual(session.seconds, 2)
        self.assertEqual(len(session.pending), 0)
        np.testing.assert_array_equal(engine.calls[0][0], samples[:16000])
        np.testing.assert_array_equal(engine.calls[1][0], samples[16000:])

        repeated_text, repeated_status = self.consume(session, np.empty(0, np.float32), engine, flush=True)
        self.assertEqual(repeated_text, text)
        self.assertNotIn("Error", repeated_status)
        self.assertEqual(len(engine.calls), 2)

    def test_empty_decode_has_status_but_no_transcript_row(self):
        session = mms.Session()
        engine = FakeEngine([mms.Result(text="", latency=0.5)])
        text, status = self.consume(session, tone(), engine)
        self.assertEqual(text, "")
        self.assertEqual(session.rows, [])
        self.assertIn("No text decoded", status)
        self.assertIn("0.50 s", status)
        self.assertEqual(session.seconds, 1)

    def test_slow_decode_reports_capture_backlog(self):
        engine = FakeEngine([mms.Result(text="Transcript", latency=1.5)])
        text, status = self.consume(mms.Session(), tone(), engine)
        self.assertIn("Decode latency: 1.50 s; audio: 1.00 s", text)
        self.assertIn("slower than capture", status)

    def test_transcript_retains_only_last_120_chunks_with_original_timestamps(self):
        session = mms.Session()
        engine = FakeEngine()
        samples = np.zeros((mms.MAX_TRANSCRIPT_ROWS + 3) * 16000, np.float32)
        text, _ = self.consume(session, samples, engine)
        self.assertEqual(mms.MAX_TRANSCRIPT_ROWS, 120)
        self.assertEqual(len(session.rows), 120)
        self.assertTrue(session.rows[0].startswith("[00:03.00–00:04.00] Chunk 4\n"))
        self.assertTrue(session.rows[-1].startswith("[02:02.00–02:03.00] Chunk 123\n"))
        self.assertNotIn("] Chunk 1\n", text)
        self.assertNotIn("] Chunk 2\n", text)
        self.assertNotIn("] Chunk 3\n", text)
        self.assertEqual(session.seconds, 123)
        self.assertEqual(len(session.pending), 0)


class MicrophoneRouteTests(unittest.IsolatedAsyncioTestCase):
    async def post_audio(self, engine, body, headers=None):
        app = Starlette(routes=mms.microphone_routes(engine))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/mms/transcribe", content=body, headers=headers)

    async def test_float_pcm_is_resampled_and_result_returned_as_json(self):
        engine = FakeEngine([mms.Result(text="Hakha Chin", latency=0.25)])
        samples = tone(sample_rate=48000)
        response = await self.post_audio(engine, samples.astype("<f4").tobytes(),
                                         {"X-Sample-Rate": "48000", "X-Threshold-DB": "-37"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "Hakha Chin", "latency": 0.25, "skipped": False})
        self.assertEqual(len(engine.calls), 1)
        normalized, threshold = engine.calls[0]
        self.assertEqual(normalized.shape, (16000,))
        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_allclose(normalized[160:-160], tone()[160:-160], atol=0.001)
        self.assertEqual(threshold, -37)

    async def test_silent_and_empty_requests_skip_before_inference(self):
        engine = mms.MMSEngine.__new__(mms.MMSEngine)
        for body in (b"", np.zeros(16000, dtype="<f4").tobytes()):
            with self.subTest(bytes=len(body)):
                response = await self.post_audio(engine, body, {"X-Sample-Rate": "16000"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"text": "", "latency": 0, "skipped": True})

    async def test_invalid_headers_return_400_without_inference(self):
        engine = FakeEngine()
        headers_to_reject = [
            {},
            *({"X-Sample-Rate": rate} for rate in ("not-a-rate", "16000.0", "7999", "192001")),
            *({"X-Sample-Rate": "16000", "X-Threshold-DB": value}
              for value in ("not-a-threshold", "-61", "-19", "nan", "inf", "-inf")),
        ]
        for headers in headers_to_reject:
            with self.subTest(headers=headers):
                response = await self.post_audio(engine, tone().astype("<f4").tobytes(), headers)
                self.assertEqual(response.status_code, 400)
                self.assertTrue(response.json()["error"])
                self.assertEqual(engine.calls, [])

    async def test_malformed_or_nonfinite_pcm_returns_400_without_inference(self):
        engine = FakeEngine()
        bodies = [b"bad", *(np.array([value], dtype="<f4").tobytes()
                            for value in (np.nan, np.inf, -np.inf))]
        for body in bodies:
            with self.subTest(body=body):
                response = await self.post_audio(engine, body, {"X-Sample-Rate": "16000"})
                self.assertEqual(response.status_code, 400)
                self.assertTrue(response.json()["error"])
                self.assertEqual(engine.calls, [])

    async def test_five_seconds_allowed_but_longer_chunk_rejected_without_inference(self):
        engine = FakeEngine()
        samples = np.zeros(16000 * 5, dtype="<f4")
        accepted = await self.post_audio(engine, samples.tobytes(), {"X-Sample-Rate": "16000"})
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(len(engine.calls[0][0]), 80000)

        rejected = await self.post_audio(engine, samples.tobytes() + b"\0\0\0\0",
                                         {"X-Sample-Rate": "16000"})
        self.assertEqual(rejected.status_code, 400)
        self.assertIn("five seconds", rejected.json()["error"])
        self.assertEqual(len(engine.calls), 1)

    async def test_backend_failure_returns_visible_json_error(self):
        engine = FakeEngine([RuntimeError("decoder failed")])
        response = await self.post_audio(engine, tone().astype("<f4").tobytes(),
                                         {"X-Sample-Rate": "16000"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"error": "RuntimeError: decoder failed"})
        self.assertEqual(len(engine.calls), 1)


if __name__ == "__main__":
    unittest.main()
