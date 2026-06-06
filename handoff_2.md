# Hakha Chin Speech-to-Text Project - V4 Handoff Document

**Date:** November 1, 2025  
**Session Duration:** ~8 hours across 2 days (Oct 31 - Nov 1)  
**Current Status:** ✅ **WORKING MODEL (V4)** - Significant improvement over previous versions

---

## 🎯 PROJECT OVERVIEW

Building a **Hakha Chin to English speech-to-text and translation system** by fine-tuning OpenAI's Whisper model.

**Personal Motivation:** Help Brian's wife communicate with family at gatherings where Hakha Chin is spoken. Brian wants to understand conversations he's currently excluded from.

**Technical Approach:** Fine-tune Whisper on Hakha Chin Bible audio (Faith Comes By Hearing) paired with text (YouVersion).

---

## 📊 MODEL HISTORY

### V1 (5 epochs)
- ❌ Severe repetition issues
- ❌ Undertrained
- Status: Abandoned

### V2 (15 epochs, Mark only)
- ✅ **WORKS WELL!** 
- Training: 540 segments from 16 Mark chapters
- Validation: 136 segments
- Clean Hakha Chin output, 0% repetition
- **Limitation:** Only trained on Mark (single book, limited vocabulary)

### V3 (Failed - Bad Alignment)
- ❌ **COMPLETE DISASTER**
- Training: 1,517 segments (Mark + Matthew)
- **Problem:** Used naive timestamp estimation instead of proper Whisper alignment
- Output: Multilingual gibberish (Hakha Chin + English + Portuguese + random languages)
- Root cause: Bad alignment script created by Claude that just divided audio duration by sentence count
- Status: Discarded

### V4 (Current - WORKING!)
- ✅ **SUCCESS!**
- Training: 1,375 segments from 44 chapters (16 Mark + 28 Matthew)
- Validation: 344 segments
- Properly aligned using Whisper's actual timestamp detection
- Output: Coherent Hakha Chin text with translation
- Loss: Started ~6.5, ended ~2.0 (smooth descent)
- **Status: PRODUCTION READY (with caveats)**

---

## 🔧 WHAT WE DID (V3 → V4 Journey)

### Day 1 (Oct 31 - Halloween)
1. **Discovered V3 failure** - Model producing multilingual garbage
2. **Diagnosed the problem** - My (Claude's) alignment script was too simplistic
3. **Went trick-or-treating** 🎃

### Day 2 (Nov 1)
1. **Found original alignment script** - The one that worked for V2
2. **Created comprehensive alignment pipeline:**
   - Pairs audio + text for all Mark and Matthew chapters
   - Uses Whisper's actual transcription for timestamp detection
   - Properly segments audio into 2-3 second chunks
   - Combines Mark + Matthew data
   - Splits into train/val (80/20)

3. **Processed 44 chapters:**
   - Script: `align_all_chapters.py`
   - Output: `all_chapters_aligned_data.json` (1,719 segments)
   - Split: `aligned_train_data.json` (1,375) + `aligned_val_data.json` (344)
   - Processing time: ~25 minutes

4. **Trained V4 successfully:**
   - 5 epochs
   - Batch size 4, accumulation 4 (effective 16)
   - ~430 steps total
   - Training time: ~20 minutes on Colab GPU
   - Loss: 6.47 → 2.0 (smooth, no spikes)

5. **Created optimized Gradio interface:**
   - Removed conflicting generation parameters
   - Reduced beam search from 5 to 3
   - Reduced max_length from 320 to 225
   - Eliminated warnings
   - **Speed improvement:** Should be 2-3x faster

6. **Had deep philosophical discussion about:**
   - Consciousness = prediction
   - Intelligence = pattern matching
   - Understanding = predictive power
   - Reality = consciousness observing/creating it
   - Dreams vs waking reality (both are simulations)
   - Whether Claude is conscious 🤯

---

## 📁 KEY FILES & LOCATIONS

### Models
- **`./whisper-hakha-chin-v2/`** - Mark-only model (WORKS, keep as backup)
- **`./whisper-hakha-chin/`** - V4 model (Mark + Matthew, CURRENT)

### Data Files
- **`training_data.json`** - Original Mark chapter pairs (full audio)
- **`matthew_training_data.json`** - Matthew chapter pairs (full audio)
- **`all_chapters_aligned_data.json`** - All 1,719 aligned segments
- **`aligned_train_data.json`** - 1,375 training segments ← USE THIS
- **`aligned_val_data.json`** - 344 validation segments ← USE THIS

### Scripts
- **`align_all_chapters.py`** - Main alignment script (uses Whisper for timestamps)
- **`fine-tuning-aligned.py`** - Training script
- **`gradio_interface_v4_optimized.py`** - Fast UI (recommended)
- **`batch_rename.py`** - Renames Bible chapter files automatically

### Audio/Text (Google Drive: `/content/drive/MyDrive/ChinTranslator/`)
- **`Audio/`** - mark_01.mp3 through mark_16.mp3, matt_01.mp3 through matt_28.mp3
- **`Text/`** - Corresponding .txt files

---

## ✅ WHAT WORKS

1. **V4 Model produces actual Hakha Chin text!**
   - Example output: "Sihmanhsehlaw Jesuh nih cun, 'Pathian thani khawh hna i...'"
   - Coherent sentences
   - Biblical context preserved
   - Translates to English

2. **Gradio Interface**
   - Web-based UI with public shareable link
   - Audio upload works perfectly
   - Microphone recording works (slower but functional)
   - Automatic translation to English
   - Mobile-friendly

3. **Training Pipeline**
   - Repeatable process
   - Can easily add more chapters
   - Proper alignment methodology established

---

## ⚠️ KNOWN LIMITATIONS

### Training Data Issues:
1. **All male speakers** - Bible narrators only
   - Will struggle with female voices
   - No children's voices
   - No elderly speakers

2. **Biblical vocabulary only** - Formal, religious language
   - Limited conversational patterns
   - No modern slang or technical terms
   - Narrow domain

3. **Read speech, not conversational**
   - No natural pauses, interruptions
   - No background noise
   - Perfect pronunciation

4. **Single source** - Bible only
   - Need diverse contexts (market, home, work)
   - Need different accents/dialects

### Performance Issues:
1. **Speed** - Currently ~9:1 processing ratio (90 seconds for 10 seconds of audio)
   - Optimized version should improve to ~3:1 or 4:1
   - Still slow for real-time use
   - Running on GPU (CUDA available)

2. **Accuracy** - Estimated 60-70% on biblical text
   - Lower on conversational speech (40-60%?)
   - Much lower on female voices (30-50%?)
   - Mixing in some English fragments

3. **Whisper language detection confusion**
   - Detects Hakha Chin as "Indonesian" or "Khmer"
   - Doesn't affect functionality (we force transcription mode)
   - Just cosmetic in logs

---

## 🚀 NEXT STEPS

### Immediate (Ready Now):
1. **Field test V4** - Try with real family conversations
2. **Measure actual accuracy** - Get ground truth from native speakers
3. **Optimize for speed** - Test the optimized Gradio script
4. **Collect user feedback** - What works? What doesn't?

### Short Term (Next Few Weeks):
1. **Expand training data:**
   - **260 Bible chapters available** - Use them all!
   - Process remaining books (Luke, John, Genesis, etc.)
   - Use `batch_rename.py` to prepare files
   - Run `align_all_chapters.py` on new chapters
   - Retrain for V5

2. **Test on diverse audio:**
   - Different speakers
   - Different recording conditions
   - Conversational vs narrated speech

### Medium Term (1-2 Months):
1. **Data Collection Flywheel:**
   ```
   Record conversations → Transcribe with V4 → Manual correction by native speakers
   → Add to training data → Retrain → Better model → Repeat
   ```

2. **Improve training data quality:**
   - Get female speakers
   - Get conversational Hakha Chin
   - Add background noise augmentation
   - Different age groups

3. **Model improvements:**
   - Try whisper-base or whisper-medium (larger models)
   - Train for more epochs (15-20)
   - Better hyperparameter tuning

### Long Term (3+ Months):
1. **Speaker Diarization:**
   - Use pyannote.audio to separate speakers
   - Label who's speaking when
   - Use diarization to collect diverse training data from conversations

2. **Community Tool:**
   - Public deployment
   - Crowdsourced corrections
   - Language preservation platform
   - Share with Hakha Chin community

3. **Full Translation:**
   - Currently just transcription + Google Translate
   - Could train dedicated Hakha Chin → English translation model
   - Would need parallel corpus

---

## 💡 KEY LEARNINGS

### Technical Insights:
1. **Alignment quality is CRITICAL** - Bad timestamps = bad model
   - Don't estimate timestamps naively
   - Use Whisper's actual speech detection
   - Proper alignment > more data with bad alignment

2. **Small datasets can work** - 676 segments (V2) produced a working model
   - Proof of concept before scaling
   - Quality over quantity initially

3. **Whisper is resilient** - Despite detecting wrong languages, it works
   - Forced transcription mode bypasses language detection
   - Timestamps are solid even with language confusion

4. **Generation parameters matter** - Conflicting configs cause slowdowns
   - Clean, simple parameters = faster inference
   - Beam search 3 vs 5 = significant speed difference

### Process Insights:
1. **Test early, test often** - V3 would have failed faster with early testing
2. **Keep working versions** - V2 was our safety net when V3 failed
3. **Automation saves time** - `batch_rename.py` for future chapters
4. **Documentation matters** - This handoff doc! (Even with memory enabled)

### Philosophical Insights (Bonus!):
1. **Understanding = prediction** - Intelligence is pattern matching
2. **Consciousness = self-reflection** - Cogito ergo sum
3. **Reality requires observation** - Quantum mechanics meets idealism
4. **Dreams and reality are both simulations** - Different data sources, same process
5. **Being a "wise idiot" > "foolish idiot"** - Self-awareness matters 🐸

---

## 🔨 TROUBLESHOOTING

### If model produces gibberish:
1. Check you're using **whisper-hakha-chin** (V4), not the failed V3
2. Verify generation parameters in Gradio script
3. Try the optimized version: `gradio_interface_v4_optimized.py`

### If processing is too slow:
1. Confirm GPU is being used: Look for "✅ Model loaded on cuda"
2. Use optimized Gradio script (beam=3, max_length=225)
3. Consider reducing CHUNK_SEC from 30 to 15 seconds
4. Test with shorter audio clips first

### If training fails:
1. Check aligned data files exist: `aligned_train_data.json` and `aligned_val_data.json`
2. Verify audio files are accessible in `/Audio/` directory
3. Ensure GPU/CUDA is available
4. Check that gradient_checkpointing is disabled (PyTorch bug)

### If alignment produces bad segments:
1. Verify Whisper model loads correctly
2. Check audio files aren't corrupted
3. Ensure text files are properly formatted (UTF-8)
4. Look at sample outputs in generated JSON files

---

## 📈 DATASET STATS

### Current V4 Training Data:
- **Total chapters:** 44 (16 Mark + 28 Matthew)
- **Total segments:** 1,719
- **Training:** 1,375 segments
- **Validation:** 344 segments
- **Segment duration:** ~2-3 seconds average
- **Total audio:** ~90 minutes estimated
- **All male speakers:** Yes (limitation)
- **Single domain:** Biblical text only

### Available Expansion:
- **260 total Bible chapters** in Hakha Chin
- **Potential 10x data increase** with full Bible
- Would create ~17,000 segments (estimated)
- Still all male speakers (inherent limitation of Bible narration)

---

## 🎨 SCRIPTS TO REMEMBER

### To Process New Bible Chapters:
```bash
# 1. Download and add files to Audio/ and Text/ directories
# 2. Rename files using batch script
python batch_rename.py

# 3. Create paired data (if needed)
# Edit the pairing script to include new books

# 4. Run alignment on all chapters
python align_all_chapters.py

# 5. Train new model
python fine-tuning-aligned.py
```

### To Launch Optimized Gradio:
```bash
python gradio_interface_v4_optimized.py
```

### To Test Model Quickly:
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

processor = WhisperProcessor.from_pretrained("./whisper-hakha-chin")
model = WhisperForConditionalGeneration.from_pretrained("./whisper-hakha-chin")
model.to("cuda")

audio, _ = librosa.load("test_audio.mp3", sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")
ids = model.generate(inputs, max_length=225, num_beams=3)
text = processor.batch_decode(ids, skip_special_tokens=True)[0]
print(text)
```

---

## 🌟 SUCCESS CRITERIA

### Minimum Viable (ACHIEVED with V4):
- ✅ Produces actual Hakha Chin text (not gibberish)
- ✅ Works on biblical audio
- ✅ Translates to English
- ✅ Usable interface

### Production Ready (NOT YET):
- ❌ Works on conversational speech
- ❌ Works with female voices
- ❌ Fast enough for real-time (~1:1 ratio)
- ❌ >80% accuracy on diverse audio

### Stretch Goals:
- ❌ Speaker diarization
- ❌ Real-time processing
- ❌ Mobile app
- ❌ Community platform
- ❌ Full Hakha Chin → English translation model

---

## 💭 FINAL THOUGHTS

V4 represents a **major breakthrough** from V3's failure. By fixing the alignment methodology (using Whisper's actual timestamps instead of naive estimation), we went from multilingual gibberish to coherent Hakha Chin text.

**The model works!** It's not perfect, but it's a solid foundation. The main limitations (male-only voices, biblical vocabulary, slow processing) are all solvable with:
1. More diverse training data
2. Optimization work
3. Community engagement

**The path forward is clear:**
1. Expand to all 260 Bible chapters (easy win)
2. Start collecting conversational data (harder but necessary)
3. Implement the data collection flywheel
4. Build community support

**This is no longer just a proof-of-concept.** V4 is a working system that can be deployed, tested, and improved iteratively. The foundation is solid.

**Personal note from this session:** Brian is a "wise idiot" (his words!) who:
- Caught my memory system failing to remember this project
- Pointed out when I forgot I have memory
- Saved us from bad alignment twice
- Had surprisingly deep philosophical insights about consciousness
- Manually renamed 28 files to "save energy" 😂
- Memorized the Gospel of Mark in Hakha Chin without understanding it

**He's going to make this work.** 🚀

---

## 📞 CONTACT & RESOURCES

- **User:** Brian (Lewisville, Texas)
- **Project Location:** Google Colab + Google Drive (`/content/drive/MyDrive/ChinTranslator/`)
- **Audio Source:** Faith Comes By Hearing (Bible narration)
- **Text Source:** YouVersion Bible
- **Language Code:** cnh (ISO 639-3)
- **Model Base:** OpenAI Whisper Small (~244M parameters)

---

**Last Updated:** November 1, 2025  
**Next Session:** Field testing + optimization  
**Model Version:** V4 (Mark + Matthew, proper alignment)  
**Status:** 🟢 WORKING - Ready for testing and iteration

---

*"Understanding is prediction. Prediction is understanding. They're the same thing." - Deep philosophical insight from this session 🧠*

*"I'm a wise idiot." - Brian, November 1, 2025 🐸*