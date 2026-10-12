"""Bounded, in-process experimental Hakha Chin -> English translation."""

import json
import threading
import time

MODEL_ID = "Heng666/madlad400-3b-mt-ct2-int8"
MODEL_REVISION = "b963b846ca4a5c106b7c64b863802599993e5f3b"
MAX_CHARACTERS = 1200
MAX_INPUT_TOKENS = 256
MAX_OUTPUT_TOKENS = 128


class TranslationBusy(RuntimeError):
    pass


class EnglishTranslator:
    """One CPU worker, INT8 weights, no external inference or transcript cache."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.tokenizer = None

    def _load(self):
        if self.model is not None:
            return
        import ctranslate2
        from huggingface_hub import snapshot_download
        from sentencepiece import SentencePieceProcessor

        path = snapshot_download(
            MODEL_ID, revision=MODEL_REVISION, token=False,
            allow_patterns=["model.bin", "config.json", "shared_vocabulary.json", "spiece.model"],
        )
        tokenizer = SentencePieceProcessor(model_file=f"{path}/spiece.model")
        model = ctranslate2.Translator(
            path, device="cpu", compute_type="int8", inter_threads=1, intra_threads=2,
        )
        self.tokenizer, self.model = tokenizer, model
        print(f"English translation ready: MADLAD-400 CPU INT8, revision={MODEL_REVISION}", flush=True)

    def translate(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Provide nonempty Hakha Chin text.")
        if len(text) > MAX_CHARACTERS:
            raise ValueError("Translation excerpt is too long; use a shorter recording.")
        if not self.lock.acquire(blocking=False):
            raise TranslationBusy("Translation is busy. Please retry shortly.")
        try:
            self._load()
            # The CT2 model has add_source_eos=false. Raw SentencePiece does
            # not add T5's EOS, so append it explicitly, matching T5Tokenizer.
            tokens = self.tokenizer.encode("<2en> " + text.strip(), out_type=str) + ["</s>"]
            if len(tokens) > MAX_INPUT_TOKENS:
                raise ValueError("Translation excerpt has too many tokens; use a shorter recording.")
            started = time.perf_counter()
            result = self.model.translate_batch(
                [tokens], beam_size=1, max_input_length=MAX_INPUT_TOKENS,
                max_decoding_length=MAX_OUTPUT_TOKENS,
            )[0]
            output = result.hypotheses[0]
            return {"text": self.tokenizer.decode(output).strip(),
                    "latency": time.perf_counter() - started,
                    "truncated": len(output) >= MAX_OUTPUT_TOKENS,
                    "experimental": True, "model": MODEL_ID, "revision": MODEL_REVISION}
        finally:
            self.lock.release()


def translation_routes(translator):
    from starlette.concurrency import run_in_threadpool
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def translate(request):
        try:
            body = bytearray()
            async for part in request.stream():
                body.extend(part)
                if len(body) > 8192:
                    return JSONResponse({"error": "Translation request too large."}, status_code=413)
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            result = await run_in_threadpool(translator.translate, payload.get("text"))
            return JSONResponse(result)
        except (ValueError, UnicodeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except TranslationBusy as exc:
            return JSONResponse({"error": str(exc)}, status_code=429)
        except Exception:
            # No transcript or raw exception is logged or returned.
            return JSONResponse({"error": "English translation unavailable. Your Hakha Chin transcript is preserved."}, status_code=503)

    return [Route("/mms/translate", translate, methods=["POST"])]
