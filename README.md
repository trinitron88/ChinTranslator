# Hakha Chin Speech-to-Text Translator

A fine-tuned Whisper model for transcribing Hakha Chin (CNH) speech to text and translating to English. Built to help bridge language barriers in Hakha Chin-speaking communities.

## 🎯 Overview

This project fine-tunes OpenAI's Whisper model on Hakha Chin Bible audio to create a speech-to-text system for this low-resource language. The model transcribes Hakha Chin audio and provides automatic translation to English.

**Current Status:** ✅ V4 Model - Production Ready (with limitations)

### Features

- **Speech-to-Text**: Transcribe Hakha Chin audio to text
- **Translation**: Automatic translation to English
- **Web Interface**: Easy-to-use Gradio interface
- **Audio Processing**: Handles uploaded files and microphone input
- **Sliding Window**: Processes long audio in manageable chunks

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- GPU recommended (CUDA support for faster processing)
- ~2GB disk space for model files

### Installation

```bash
# Clone the repository
git clone https://github.com/trinitron88/ChinTranslator2.git
cd ChinTranslator2

# Install dependencies
pip install torch transformers gradio librosa deep-translator soundfile numpy
```

### Running the Interface

```bash
python gradio_interface.py
```

The interface will launch in your browser with a shareable public link.

## 📊 Model Performance

### V4 Model (Current)
- **Training Data**: 1,375 segments from 44 Bible chapters (Mark & Matthew)
- **Validation Data**: 344 segments
- **Training Loss**: 6.47 → 2.0 (smooth descent)
- **Estimated Accuracy**: 60-70% on biblical text

### Known Limitations

1. **Training Data Constraints**:
   - All male narrators (Bible speakers only)
   - Biblical/formal vocabulary domain
   - Read speech, not conversational
   - Single audio source

2. **Performance**:
   - Processing speed: ~3-4x real-time on GPU
   - Lower accuracy on non-biblical conversational speech
   - Reduced accuracy on female voices

3. **Domain**:
   - Best for biblical or formal Hakha Chin
   - Limited modern/conversational vocabulary

## 🔧 Project Structure

```
.
├── README.md
├── gradio_interface.py           # Main web interface (optimized)
├── fine-tuning-aligned.py        # Training script
├── whisper_alignment_2.py        # Audio-text alignment
├── process-matthew.py            # Data preprocessing
├── continue_training.py          # Continue training existing model
├── aligned_train_data.json       # Training segments (1,375)
├── aligned_val_data.json         # Validation segments (344)
├── Audio/                        # Audio files (mark_*.mp3, matt_*.mp3)
├── Text/                         # Text transcripts (*.txt)
└── whisper-hakha-chin/          # Fine-tuned model (V4)
```

## 📚 Usage

### Web Interface

1. Launch the Gradio interface:
   ```bash
   python gradio_interface.py
   ```

2. Choose input method:
   - **Upload Audio**: Upload an audio file (MP3, WAV, etc.)
   - **Record Audio**: Use your microphone to record

3. Click "Transcribe & Translate"

4. View results:
   - Hakha Chin transcription
   - English translation

### Training a New Model

1. Prepare your data:
   - Audio files in `Audio/` directory
   - Corresponding text files in `Text/` directory
   - Use naming convention: `book_chapter.mp3` and `book_chapter.txt`

2. Align audio and text:
   ```bash
   python whisper_alignment_2.py
   ```

3. Train the model:
   ```bash
   python fine-tuning-aligned.py
   ```

4. Model will be saved to `./whisper-hakha-chin/`

## 🛠️ Technical Details

### Model Architecture
- **Base Model**: OpenAI Whisper Small (244M parameters)
- **Task**: Transcription (not translation)
- **Language**: Hakha Chin (forced, no language token)
- **Approach**: Fine-tuning with frozen encoder, trainable decoder

### Training Configuration
- **Epochs**: 5
- **Batch Size**: 4 (effective 16 with gradient accumulation)
- **Learning Rate**: 1e-5
- **Optimizer**: AdamW
- **Mixed Precision**: FP16 (on GPU)

### Audio Processing
- **Sample Rate**: 16kHz (mono)
- **Segmentation**: Non-silence detection
- **Window**: 30-second sliding windows with overlap
- **Normalization**: Automatic volume adjustment

### Translation
- **Method**: Google Translate API (via deep-translator)
- **Source**: Hakha Chin (CNH) or auto-detect
- **Target**: English

## 🔄 Model Versions

| Version | Chapters | Segments | Status | Notes |
|---------|----------|----------|--------|-------|
| V1 | Mark | - | ❌ Abandoned | Severe repetition, undertrained |
| V2 | Mark (16) | 540 | ✅ Working | Good baseline, limited vocabulary |
| V3 | Mark + Matthew | 1,517 | ❌ Failed | Bad alignment, multilingual gibberish |
| V4 | Mark + Matthew (44) | 1,375 | ✅ **Current** | Proper alignment, production ready |

## 🚀 Future Improvements

### Immediate
- Field test with native speakers
- Collect accuracy metrics on real conversations
- Optimize processing speed

### Short Term
- Expand to all 260+ available Bible chapters
- Add data augmentation (speed, pitch, noise)
- Test on diverse audio conditions

### Long Term
- Collect conversational Hakha Chin data
- Add female and diverse speakers
- Implement speaker diarization
- Train dedicated Hakha Chin → English translation model
- Create community crowdsourcing platform

## 📖 Data Sources

- **Audio**: Faith Comes By Hearing (Hakha Chin Bible)
- **Text**: YouVersion Bible (Hakha Chin)
- **Books**: Gospel of Mark (16 chapters), Gospel of Matthew (28 chapters)

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional training data (conversational Hakha Chin)
- Performance optimizations
- Accuracy improvements
- UI/UX enhancements
- Documentation

## 📝 License

This project is for educational and language preservation purposes. Please respect the licenses of:
- OpenAI Whisper (Apache 2.0)
- Bible audio and text sources
- Transformers library (Apache 2.0)

## 🙏 Acknowledgments

- OpenAI for the Whisper model
- Faith Comes By Hearing for Hakha Chin Bible audio
- YouVersion for Hakha Chin Bible text
- The Hakha Chin community

## 📧 Contact

For questions or collaboration: [GitHub Issues](https://github.com/trinitron88/ChinTranslator2/issues)

---

**Last Updated**: November 5, 2025  
**Model Version**: V4  
**Status**: Production Ready (with known limitations)
