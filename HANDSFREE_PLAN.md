# Hands-free, screen-off Hakha Chin → English interpreter — research & plan

**Goal:** earbud in, phone in pocket, screen OFF / app backgrounded, continuous
mic → English in your ear. The current FastRTC HF Space (`bsantisi/chin-realtime`)
works only with the browser tab foregrounded and the phone unlocked, because
mobile browsers suspend background tabs and cut mic access on lock.

This document is **research + recommendation only** — no app is built yet.

---

## TL;DR recommendation

1. **The background-mic goal is fundamentally a native-app problem.** On iOS,
   web/PWA mic capture is killed the moment the tab is backgrounded or the
   screen locks. There is no workaround. So **path 2 (PWA) is a dead end for the
   stated goal** and should be dropped except as the "screen-on" fallback you
   already have.

2. **Build a thin native app** (recommend **Flutter**, or React Native) that
   does *only* mic-in / earbud-out using the OS background-audio facilities
   (iOS `audio` background mode + `AVAudioSession`; Android microphone-type
   **foreground service**). **Keep the existing HF Space backend unchanged for
   v1.** This is the fastest route to a true pocket experience and reuses 100%
   of the working pipeline. Effort: ~2–4 weeks to a TestFlight/Play-internal
   build.

3. **The on-device dream (path 3) is real for STT and TTS but blocked on
   translation.** whisper.cpp runs the Chin model near-real-time on a modern
   phone, and Piper does sub-second English TTS on-device. **But Hakha Chin
   (`cnh`) is NOT in NLLB-200** (only its cousin Mizo `lus_Latn` is), and your
   translation step currently depends on **Google Translate's online `sl=cnh`**.
   So "fully offline, no server" requires first *building* an offline `cnh→en`
   MT model that does not exist off the shelf. Treat on-device as **v3**, phased
   in behind the native shell, with translation kept as a tiny online text call
   until/unless an offline MT model is trained.

**Sequencing:** native shell + existing backend (v2) → migrate STT to
whisper.cpp on-device + TTS to Piper, translation still a ~1 KB online text hop
(v2.5, kills almost all GPU cost) → train offline `cnh→en` MT for true offline
(v3).

---

## Why the current setup fails background

The architecture is: **phone browser = dumb mic + speaker**, everything else on
the Space GPU (`faster-whisper` Chin model → Google Translate `sl=cnh` → gTTS),
streamed over WebRTC/FastRTC. The phone half lives entirely inside a browser
tab, and:

- **iOS Safari** suspends `getUserMedia` / `AudioContext` / WebRTC media when the
  tab is backgrounded or the screen locks. `AudioContext` is documented to stop
  even when a desktop Safari window is merely sent to the background
  ([WebKit #231105](https://bugs.webkit.org/show_bug.cgi?id=231105)); on iOS the
  suspension on lock is harder. Installing it as a PWA does **not** lift this —
  standalone PWAs have their own long-standing mic bugs
  ([WebKit #185448](https://bugs.webkit.org/show_bug.cgi?id=185448)).
- **Android** Chrome is more lenient but still throttles/suspends background
  tabs and is not reliable for all-day capture.

There is no web API on iOS that grants background mic. The capability is gated to
native apps holding the right entitlement.

---

## Option comparison

| Axis | 1. Native app (+ existing HF backend) | 2. PWA / web workaround | 3. Fully on-device |
|---|---|---|---|
| **Solves screen-off mic?** | ✅ Yes (the whole point) | ❌ No on iOS; flaky on Android | ✅ Yes (native shell required anyway) |
| **Effort** | Medium — ~2–4 wks to internal build | Low, but doesn't meet the goal | High — model conversion + **new MT model** + app |
| **Per-user GPU cost** | Unchanged (Space stays) | Unchanged | ~Zero after translation moves off-server |
| **Latency** | Same 2–5 s (network + GPU) | Same | Lower (no round-trip), but bounded by phone CPU |
| **Offline capable?** | ❌ Needs network | ❌ | ⚠️ STT/TTS yes; **`cnh` translation is the blocker** |
| **App-store release?** | Yes (TestFlight/internal first; review OK) | None | Yes |
| **Reuses current pipeline?** | 100% | 100% | STT model reusable; translate + TTS replaced |
| **Battery/thermal** | Low (phone is a thin client) | Low | Higher — continuous whisper on CPU/NPU all day |

---

## Path 1 — Native app, backend stays the HF Space (RECOMMENDED for v2)

The phone keeps being a thin client; we just swap the *browser tab* for a
*native audio shell* that the OS allows to keep the mic open in the background.

**iOS:** enable the `audio` value in `UIBackgroundModes`, configure
`AVAudioSession` with the `.record` (or `.playAndRecord`) category and activate
it. With the audio background mode, the OS keeps the session (and mic) alive with
the screen off — the same mechanism voice recorders and call apps use. Apple's
docs are explicit that the audio-session category is what lets audio "persist
when the device locks." Review is fine for a legitimate interpreter/transcription
use case, but the background-audio entitlement *will* draw reviewer attention —
the app must visibly, obviously use live audio (it does).

**Android:** declare a **microphone-type foreground service**
(`FOREGROUND_SERVICE_MICROPHONE` + `foregroundServiceType="microphone"` in the
manifest) and hold the `RECORD_AUDIO` runtime permission. Since **Android 14
(API 34)** the service *type* is mandatory and the service must be **started
while the app is in the foreground** (you can't cold-start mic capture from the
background) — so the user opens the app once and taps start; capture then
survives backgrounding and lock. A persistent notification is required and is
fine.

**Cross-platform:** **Flutter** (`audio_service` / `flutter_sound` style
plugins) or **React Native** (`react-native-track-player` background mode,
foreground-service libs) both expose exactly these primitives and let one
codebase cover both stores. Recommend **Flutter** for the cleanest background-
audio + foreground-service story, but RN is equally viable if the team prefers
JS. The networking half is trivial: the native app opens the same WebRTC session
the browser does today (FastRTC has a documented client protocol), or — simpler —
**drop WebRTC entirely** and stream PCM chunks to a small WebSocket endpoint on
the Space, since we control both ends and no longer need the browser. A WebSocket
of 16 kHz PCM frames removes the TURN/NAT pain described in `REALTIME.md`.

**Backend:** unchanged — Chin whisper → Google `sl=cnh` → gTTS/Piper on the GPU.

**Effort:** ~2–4 weeks. **Cost:** Apple Developer Program $99/yr, Google Play
$25 one-time; GPU cost unchanged. **Payoff:** actually achieves the goal.

---

## Path 2 — PWA / web background audio (NOT viable for the goal)

Assessed and rejected for the stated requirement. On iOS there is no API for
background mic; screen-lock kills the stream regardless of PWA install. Android
PWAs can sometimes keep audio alive but not reliably all day, and still can't
match a foreground-service mic. **Keep the existing web app as the screen-on,
zero-install fallback** (great for "try it in 10 seconds on a laptop"), but it
cannot be the pocket experience.

---

## Path 3 — On-device (the "own it, offline, no GPU bill" endgame, v3)

Native shell is required here too (same as Path 1), then move the ML onto the
phone piece by piece:

- **STT — feasible now.** whisper.cpp runs Whisper small near-real-time and
  medium near-real-time on modern iPhones; there are working real-time Android
  streaming demos. Your `turbo`-class fine-tune is in that range. **Conversion
  caveat:** the deployed model `bsantisi/whisper-cnh-turbo-ct2` is a
  **CTranslate2** artifact, which whisper.cpp can't read. You must convert from
  the **pre-CT2 Hugging Face transformers checkpoint** using
  `whisper.cpp/models/convert-h5-to-ggml.py` → GGML/GGUF, then quantize
  (q5/q8). Make sure the original fine-tuned checkpoint (not just the CT2 export)
  is archived; if only the CT2 model survives, re-export from the training
  checkpoint. Note current scripts can choke on `safetensors`-only checkpoints
  ([whisper.cpp #3316](https://github.com/ggml-org/whisper.cpp/issues/3316)).
- **TTS — feasible now.** Piper does real-time English TTS on-device (it targets
  Raspberry Pi 4); Android plugins and iOS Piper apps exist. Drop-in replacement
  for gTTS, removes a network round-trip, lowers latency. (This is already the
  planned v2.1 swap in `REALTIME.md`.)
- **Translation — the blocker.** This is where "no server" breaks. `cnh` is
  **absent from NLLB-200 / FLORES-200** (only Mizo `lus_Latn` is present), and
  M2M-100 doesn't cover it either. Your only turnkey `cnh→en` engine is
  **Google Translate online**. Options, worst-to-best for offline:
  1. **Keep translation online** even when STT/TTS are local — it's ~1 KB of
     text per utterance, works on the weakest connection, and already eliminates
     ~all GPU cost. **This is the pragmatic v2.5.** Not "offline," but cheap and
     low-latency.
  2. **Train an offline `cnh→en` model** by fine-tuning NLLB-200-distilled-600M
     (borrowing the related-language prior) or a small seq2seq on a Chin↔English
     parallel corpus. You already have a **Bible corpus** in the repo
     (`archive/bible_dataset/`, Mark/Matthew/James) — Bible verses are
     verse-aligned across languages, so a `cnh↔en` parallel set is bootstrappable
     from it, *but* it's narrow-domain (religious register) and small; expect it
     to underperform Google on conversational speech without much more data.
     This is a real research sub-project, not a checkbox.

**Battery/thermal:** continuous on-device whisper all day is the main downside —
sustained CPU/NPU load drains battery and heats the phone. Mitigate with VAD
gating (only run whisper on detected speech), the smallest model that holds
accuracy, and NPU/CoreML/NNAPI acceleration.

---

## Recommended roadmap

1. **v2 — Native thin client, existing backend.** Flutter (or RN) app: mic in
   background via iOS audio mode / Android mic foreground service; stream PCM to
   a small WebSocket on the Space (retire browser WebRTC/TURN). Ship to
   TestFlight + Play internal testing. *Delivers the pocket experience.*
2. **v2.1 — Piper TTS** on the backend (or on-device) for sub-second, snappier
   English. Tune VAD/chunking; partial results.
3. **v2.5 — STT on-device** (whisper.cpp GGML from the original checkpoint) +
   **Piper on-device**; **translation stays a tiny online text call.** Kills
   nearly all GPU cost, cuts latency, works on weak connections. Backend shrinks
   to almost nothing.
4. **v3 — Offline `cnh→en` MT** trained from Bible + any further parallel data →
   true no-server, offline interpreter. Gated on data quality; ship only if it
   beats the online hop for real conversation.

## Immediate next steps (when we start building)

- [ ] Decide Flutter vs React Native (recommend Flutter).
- [ ] Confirm the **original fine-tuned HF transformers checkpoint** (pre-CT2)
      is archived — it's required for the eventual whisper.cpp conversion.
- [ ] Add a WebSocket PCM endpoint to the Space alongside (or replacing) FastRTC,
      so the native client doesn't need TURN.
- [ ] Apple Developer ($99/yr) + Google Play ($25) accounts; reserve app id /
      bundle id.
- [ ] Prototype the background-audio shell on **one** platform first (iOS is the
      stricter gatekeeper — prove background capture there before Android).

---

### Sources
- iOS PWA/Safari mic limits: [MagicBell PWA iOS limits](https://www.magicbell.com/blog/pwa-ios-limitations-safari-support-complete-guide), [WebKit #185448](https://bugs.webkit.org/show_bug.cgi?id=185448), [WebKit #231105](https://bugs.webkit.org/show_bug.cgi?id=231105)
- iOS background audio: [Apple AVAudioSession](https://developer.apple.com/documentation/avfaudio/avaudiosession)
- Android 14 mic foreground service: [Android FGS types](https://developer.android.com/develop/background-work/services/fgs/service-types), [FGS background-start restrictions](https://developer.android.com/develop/background-work/services/fgs/restrictions-bg-start)
- whisper.cpp on mobile: [whisper.cpp](https://github.com/ggml-org/whisper.cpp), [Android streaming demo](https://github.com/liam-mceneaney/androidwhisper.cpp), [conversion #3316](https://github.com/ggml-org/whisper.cpp/issues/3316)
- Piper on-device: [Piper TTS](https://github.com/rhasspy/piper), [iOS Piper](https://speechcentral.net/2026/04/07/offline-system-voices-on-ios-are-finally-becoming-practical/)
- `cnh` absent from NLLB-200: [FLORES-200 language list](https://github.com/facebookresearch/flores/blob/main/flores200/README.md), [NLLB](https://huggingface.co/docs/transformers/model_doc/nllb)
- Cross-platform background audio: [Flutter audio_service](https://pub.dev/packages/audio_service), [RN Track Player background mode](https://rntp.dev/docs/basics/background-mode)
