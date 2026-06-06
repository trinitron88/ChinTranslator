import json
import librosa
import IPython.display as ipd

# Load training data
with open('train_data_fixed.json', 'r') as f:
    train_data = json.load(f)

# Pick a random segment to check
import random
segment = random.choice(train_data)

print("="*60)
print("CHECKING SEGMENT ALIGNMENT")
print("="*60)
print(f"Book: {segment['book']}")
print(f"Chapter: {segment['chapter']}")
print(f"Time: {segment['start']:.2f}s → {segment['end']:.2f}s ({segment['duration']:.2f}s)")
print(f"\nText: {segment['text']}")
print("="*60)

# Load and play the audio segment
audio, sr = librosa.load(
    segment['audio'],
    sr=16000,
    offset=segment['start'],
    duration=segment['duration']
)

print("\n🎧 Playing audio segment...")
ipd.display(ipd.Audio(audio, rate=sr))