"""
Continue Training from Checkpoint
Resume training with more epochs to fix repetition issue
"""

import torch
import json
from datasets import Dataset
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

# Configuration - INCREASED TRAINING
MODEL_NAME = "openai/whisper-small"
CHECKPOINT_PATH = "./whisper-hakha-chin"  # Your previous checkpoint
OUTPUT_DIR = "./whisper-hakha-chin-v2"     # New output directory
LANGUAGE = None
TASK = "transcribe"

print("="*50)
print("CONTINUING TRAINING - Hakha Chin Whisper")
print("Version 2: More Epochs + Anti-Repetition")
print("="*50)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

# Load data
print("\n📂 Loading aligned data...")
with open('aligned_train_data.json', 'r') as f:
    train_data = json.load(f)
with open('aligned_val_data.json', 'r') as f:
    val_data = json.load(f)

print(f"Train segments: {len(train_data)}")
print(f"Validation segments: {len(val_data)}")

# Create datasets
def prepare_dataset_metadata(data_list):
    return Dataset.from_dict({
        "audio_path": [item['audio'] for item in data_list],
        "start_time": [item['start'] for item in data_list],
        "end_time": [item['end'] for item in data_list],
        "text": [item['text'] for item in data_list]
    })

print("\n🔄 Creating dataset objects...")
train_dataset = prepare_dataset_metadata(train_data)
val_dataset = prepare_dataset_metadata(val_data)

# Load model from checkpoint (continue training)
print(f"\n🤖 Loading model from checkpoint: {CHECKPOINT_PATH}")
try:
    feature_extractor = WhisperFeatureExtractor.from_pretrained(CHECKPOINT_PATH)
    tokenizer = WhisperTokenizer.from_pretrained(CHECKPOINT_PATH, task=TASK)
    processor = WhisperProcessor.from_pretrained(CHECKPOINT_PATH, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(CHECKPOINT_PATH)
    print("✅ Loaded from checkpoint - continuing training!")
except:
    print("⚠️  Checkpoint not found, loading fresh model...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, task=TASK)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False

# Set anti-repetition parameters on model config
from transformers import GenerationConfig
generation_config = GenerationConfig.from_pretrained(CHECKPOINT_PATH if CHECKPOINT_PATH else MODEL_NAME)
generation_config.repetition_penalty = 1.5
generation_config.no_repeat_ngram_size = 3
model.generation_config = generation_config

print("✅ Model ready (with anti-repetition settings)")

# Prepare data
def prepare_data(batch):
    import librosa
    import numpy as np
    
    try:
        audio, sr = librosa.load(
            batch["audio_path"],
            sr=16000,
            offset=batch["start_time"],
            duration=batch["end_time"] - batch["start_time"]
        )
    except:
        duration = batch["end_time"] - batch["start_time"]
        audio = np.zeros(int(duration * 16000))
    
    batch["input_features"] = feature_extractor(
        audio, 
        sampling_rate=16000
    ).input_features[0]
    
    # IMPORTANT: Truncate text to avoid repetition issues
    encoded = tokenizer(batch["text"], truncation=True, max_length=200)  # Shorter max length
    batch["labels"] = encoded.input_ids
    
    return batch

print("\n🔄 Processing datasets...")
train_dataset = train_dataset.map(
    prepare_data,
    remove_columns=["audio_path", "start_time", "end_time", "text"],
    desc="Processing training data"
)
val_dataset = val_dataset.map(
    prepare_data,
    remove_columns=["audio_path", "start_time", "end_time", "text"],
    desc="Processing validation data"
)
print("✅ Data ready")

# Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Evaluation metric
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}

# IMPROVED Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,          # Increased batch size
    gradient_accumulation_steps=2,           # Effective batch = 16
    learning_rate=5e-6,                      # Slightly lower LR for fine-tuning
    warmup_steps=50,
    num_train_epochs=15,                     # MORE EPOCHS (was 5)
    gradient_checkpointing=False,
    fp16=True if device == "cuda" else False,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=200,               # Shorter to prevent loops
    save_steps=100,                          # Save more frequently
    eval_steps=100,
    logging_steps=20,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    save_total_limit=3,
    dataloader_num_workers=2,
)

# Progress callback
class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step {state.global_step}: {logs}")

# Initialize trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
    callbacks=[ProgressCallback()],
)

print("\n🚀 RESUMING TRAINING...")
print("="*50)
print(f"📊 Training on {len(train_dataset)} segments")
print(f"📊 Validating on {len(val_dataset)} segments")
print(f"📊 Epochs: 15 (was 5)")
print(f"📊 Steps per epoch: ~{len(train_dataset) // 16}")
print(f"📊 Total steps: ~{(len(train_dataset) // 16) * 15}")
print(f"📊 Anti-repetition: ENABLED")
print("="*50)

# Train!
try:
    trainer.train()
    
    print("\n✅ Training complete!")
    print(f"📁 Model saved to: {OUTPUT_DIR}")
    
    # Save final model
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print("\n🎉 Version 2 complete!")
    print("Next: Test with the Gradio interface!")
    
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted")
    print("Saving checkpoint...")
    trainer.save_model(f"{OUTPUT_DIR}/interrupted")
    processor.save_pretrained(f"{OUTPUT_DIR}/interrupted")
    print("✅ Checkpoint saved!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    try:
        trainer.save_model(f"{OUTPUT_DIR}/error")
        processor.save_pretrained(f"{OUTPUT_DIR}/error")
        print("✅ Emergency checkpoint saved!")
    except:
        pass
    raise