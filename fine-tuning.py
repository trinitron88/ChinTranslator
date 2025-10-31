"""
Whisper Fine-tuning Script for Hakha Chin
Run this in Google Colab or on a machine with GPU
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
MODEL_NAME = "openai/whisper-small"  # Start with small model
LANGUAGE = None  # Don't specify language - let model learn Hakha Chin
TASK = "transcribe"
OUTPUT_DIR = "./whisper-hakha-chin"

print("="*50)
print("Whisper Fine-tuning for Hakha Chin")
print("="*50)

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")
if device == "cpu":
    print("⚠️  WARNING: No GPU detected. Training will be very slow!")
    print("   Consider using Google Colab with GPU enabled.")

# Load the data
print("\n📂 Loading data...")
with open('./train_data.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)
with open('./val_data.json', 'r', encoding='utf-8') as f:
    val_data = json.load(f)

print(f"Train samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

# Convert to HuggingFace Dataset format
def prepare_dataset(data_list):
    return Dataset.from_dict({
        "audio": [item["audio"] for item in data_list],
        "text": [item["text"] for item in data_list]
    })

train_dataset = prepare_dataset(train_data)
val_dataset = prepare_dataset(val_data)

# Cast audio column to Audio feature
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16000))

print("✅ Datasets prepared")

# Load Whisper model components
print("\n🤖 Loading Whisper model...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, task=TASK)
processor = WhisperProcessor.from_pretrained(MODEL_NAME, task=TASK)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Enable gradient checkpointing to save memory
model.config.use_cache = False
model.gradient_checkpointing_enable()

print("✅ Model loaded")

# Prepare data for training
def prepare_data(batch):
    # Load audio
    audio = batch["audio"]
    
    # Compute input features
    batch["input_features"] = feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Encode target text
    batch["labels"] = tokenizer(batch["text"]).input_ids
    
    return batch

print("\n🔄 Processing datasets...")
train_dataset = train_dataset.map(
    prepare_data, 
    remove_columns=train_dataset.column_names
)
val_dataset = val_dataset.map(
    prepare_data, 
    remove_columns=val_dataset.column_names
)
print("✅ Datasets processed")

# Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split into input features and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding with -100 (ignore in loss)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        # Remove BOS token if present
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
    
    # Replace -100 with pad token
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,  # Adjust based on GPU memory
    gradient_accumulation_steps=2,  # Effective batch size = 8
    learning_rate=1e-5,
    warmup_steps=50,
    num_train_epochs=10,  # Adjust as needed
    gradient_checkpointing=True,
    fp16=True if device == "cuda" else False,  # Mixed precision training
    eval_strategy="steps",  # Changed from evaluation_strategy
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

# Custom callback for progress
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
    tokenizer=processor.feature_extractor,
    callbacks=[ProgressCallback()],
)

print("\n🚀 Starting training...")
print("="*50)

# Train!
trainer.train()

print("\n✅ Training complete!")
print(f"📁 Model saved to: {OUTPUT_DIR}")

# Save final model
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("\n🎉 All done! Your fine-tuned Hakha Chin Whisper model is ready!")
print(f"\nTo use it:")
print(f"  model = WhisperForConditionalGeneration.from_pretrained('{OUTPUT_DIR}')")
print(f"  processor = WhisperProcessor.from_pretrained('{OUTPUT_DIR}')")