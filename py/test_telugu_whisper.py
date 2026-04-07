#!/usr/bin/env python3
"""
Test script to verify Telugu Whisper model works correctly
"""
import torch
from transformers import pipeline
import os

def test_telugu_whisper():
    print("Testing Telugu Whisper model...")
    
    # Check if model exists in cache
    model_path = "/home/ssp/ML/proejct/py/.model_cache/models--vanshnawander--whisper-small-telugu"
    if not os.path.exists(model_path):
        print("Model not found in cache. Please run setup_models.py first.")
        return
    
    # Find the actual model directory
    snapshots_dir = os.path.join(model_path, "snapshots")
    if os.path.exists(snapshots_dir):
        snapshot_dirs = os.listdir(snapshots_dir)
        if snapshot_dirs:
            model_dir = os.path.join(snapshots_dir, snapshot_dirs[0])
            print(f"Using model from: {model_dir}")
        else:
            print("No snapshot found")
            return
    else:
        model_dir = model_path
    
    try:
        # Create ASR pipeline
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model_dir,
            device=0 if torch.cuda.is_available() else -1,
        )
        
        print("Model loaded successfully!")
        print("Model is ready for Telugu transcription.")
        
        # Test with a dummy audio (just to verify loading works)
        print("\nTo test with actual Telugu audio:")
        print("1. Place a Telugu audio file in the project directory")
        print("2. Use: result = asr_pipe('path_to_telugu_audio.wav', generate_kwargs={'language': 'te', 'task': 'transcribe'})")
        print("3. Print result['text'] for transcription")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_telugu_whisper()