"""
Whisper Fine-tuning Script for Hakha Chin - FIXED VERSION
Fixes the gradient checkpointing bug
Run this in Google Colab with GPU
"""

import torch
import json
from pathlib import Path
from datasets import Dataset, Audio
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

# Configuration
MODEL_NAME = "openai/whisper-small"
LANGUAGE = None
TASK = "transcribe"
OUTPUT_DIR = "./whisper-hakha-chin"

print("="*50)
print("Whisper Fine-tuning for Hakha Chin")
print("FIXED VERSION - No Gradient Checkpointing Bug")
print("="*50)

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")
if device == "cpu":
    print("⚠️  WARNING: No GPU detected. Training will be very slow!")

# Load the ALIGNED data
print("\n📂 Loading aligned data...")
with open('combined_train_data.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)
with open('combined_val_data.json', 'r', encoding='utf-8') as f:
    val_data = json.load(f)

print(f"Train segments: {len(train_data)}")
print(f"Validation segments: {len(val_data)}")

# Show sample
print("\n📋 Sample training item:")
sample = train_data[0]
print(f"  Audio file: {sample['audio']}")
print(f"  Start time: {sample['start']:.2f}s")
print(f"  End time: {sample['end']:.2f}s")
print(f"  Text: {sample['text'][:100]}...")

# Create datasets with metadata only
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
print("✅ Dataset metadata ready")

# Load Whisper model components
print("\n🤖 Loading Whisper model...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, task=TASK)
processor = WhisperProcessor.from_pretrained(MODEL_NAME, task=TASK)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# CRITICAL FIX: Disable gradient checkpointing to avoid the backward bug
model.config.use_cache = False
# DO NOT enable gradient checkpointing - it causes the error
# model.gradient_checkpointing_enable()  # COMMENTED OUT

print("✅ Model loaded (gradient checkpointing disabled)")

# Prepare data for training
def prepare_data(batch):
    import librosa
    import numpy as np
    
    # Load audio segment
    try:
        audio, sr = librosa.load(
            batch["audio_path"],
            sr=16000,
            offset=batch["start_time"],
            duration=batch["end_time"] - batch["start_time"]
        )
    except Exception as e:
        # If audio loading fails, use silence
        print(f"Warning: Failed to load audio, using silence: {e}")
        duration = batch["end_time"] - batch["start_time"]
        audio = np.zeros(int(duration * 16000))
    
    # Compute input features
    batch["input_features"] = feature_extractor(
        audio, 
        sampling_rate=16000
    ).input_features[0]
    
    # Encode target text - TRUNCATE if too long
    encoded = tokenizer(batch["text"], truncation=True, max_length=448)
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
print("✅ Data pipeline ready")

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

# Training arguments - FIXED
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=100,
    num_train_epochs=5,
    gradient_checkpointing=False,  # DISABLED to fix the bug
    fp16=True if device == "cuda" else False,
    eval_strategy="steps",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=200,
    eval_steps=200,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    save_total_limit=2,
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
    processing_class=processor.feature_extractor,  # Updated from 'tokenizer'
    callbacks=[ProgressCallback()],
)

print("\n🚀 Starting training...")
print("="*50)
print(f"📊 Training on {len(train_dataset)} segments")
print(f"📊 Validating on {len(val_dataset)} segments")
print(f"📊 Batch size: 4, Accumulation: 4 (Effective: 16)")
print(f"📊 Total steps: ~{(len(train_dataset) // 16) * training_args.num_train_epochs}")
print("="*50)

# Train!
try:
    trainer.train()
    
    print("\n✅ Training complete!")
    print(f"📁 Model saved to: {OUTPUT_DIR}")
    
    # Save final model
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print("\n🎉 All done! Your fine-tuned Hakha Chin Whisper model is ready!")
    
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
    print("Saving current checkpoint...")
    trainer.save_model(f"{OUTPUT_DIR}/interrupted_checkpoint")
    processor.save_pretrained(f"{OUTPUT_DIR}/interrupted_checkpoint")
    print("✅ Checkpoint saved!")

except Exception as e:
    print(f"\n❌ Error during training: {e}")
    print("Attempting to save checkpoint...")
    try:
        trainer.save_model(f"{OUTPUT_DIR}/error_checkpoint")
        processor.save_pretrained(f"{OUTPUT_DIR}/error_checkpoint")
        print("✅ Emergency checkpoint saved!")
    except:
        print("❌ Could not save checkpoint")
    raise