"""CPU-only Hugging Face Space entrypoint for the isolated MMS experiment.

Importing this module does not download weights or start a server. Export the
Space with prepare_space.py so the shared mms_realtime.py travels with it.
"""

import os
import sys
from pathlib import Path

# Set before importing Gradio or Hugging Face packages. No token is needed for
# the public MMS model; this app never provisions or upgrades Space hardware.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

HERE = Path(__file__).resolve().parent
CORE_DIR = HERE if (HERE / "mms_realtime.py").is_file() else HERE.parent
if not (CORE_DIR / "mms_realtime.py").is_file():
    raise RuntimeError("Missing mms_realtime.py. Export the Space with prepare_space.py first.")
sys.path.insert(0, str(CORE_DIR))

import gradio as gr

from mms_realtime import MMSEngine, build_demo, microphone_routes
from mms_translation import EnglishTranslator, translation_routes

HOSTING_NOTICE = """
**Hosted experiment:** This version processes microphone audio on the Hugging
Face Space's server. It uses CPU inference and may fall behind live speech.
It attempts English translation of the latest four transcribed chunks after Stop.
It does not generate speech. Transcription and translation accuracy have not been evaluated. Start with a short, consented, non-sensitive test.

**Privacy on this host:** Audio leaves your device over HTTPS and goes to this
Space's backend. This app does not intentionally save recordings or transcript
history to disk, and does not send audio or transcript text to a separate inference API. Hosting
providers may retain ordinary network/service logs; this is not a guarantee of
zero retention. Gradio analytics and Hub telemetry are disabled. No Gradio
share tunnel is started. Clear stops capture and discards waiting browser
chunks; inference already running on the server may still finish. Refresh or
close the page when done. Do not use for confidential conversations.
"""


def build_space(engine):
    """Reuse the existing recorder and show hosting-specific privacy context."""
    demo = build_demo(engine, chunk_seconds=3.0, english_translation=True)
    with demo:
        gr.Markdown(HOSTING_NOTICE)
    return demo


def launch_space(engine=None, *, server_name="0.0.0.0", server_port=7860,
                 prevent_thread_lock=False):
    """Launch the UI AND its custom, in-memory PCM endpoint on one server.

    An injected engine is only a local testing seam. Normal execution always
    loads the real MMS cnh adapter on CPU, with no fallback to a different model.
    """
    if engine is None:
        engine = MMSEngine(device="cpu")
    translator = EnglishTranslator()
    demo = build_space(engine)
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        # Spaces may enable SSR by environment. Gradio 5.23.1 then proxies
        # this custom PCM route to its Node frontend rather than Python.
        ssr_mode=False,
        show_error=False,
        prevent_thread_lock=prevent_thread_lock,
        app_kwargs={"routes": microphone_routes(engine) + translation_routes(translator)},
    )
    return demo


if __name__ == "__main__":
    launch_space()
