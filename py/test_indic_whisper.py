#!/usr/bin/env python3
"""
Test script to verify Indic Whisper model works correctly
"""
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import os

def test_indic_whisper():
    print("Testing Indic Whisper model...")
    
    # Check if model exists in cache
    model_path = "/home/ssp/ML/proejct/py/.model_cache/models--parthiv11--indic_whisper_nodcil"
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
        # Load model and processor
        print("Loading processor...")
        processor = WhisperProcessor.from_pretrained(model_dir)
        print("Loading model...")
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Processor type: {type(processor)}")
        print("Model is ready for Indian language transcription.")
        
        # Test with a dummy audio (just to verify loading works)
        print("\nTo test with actual Indian language audio:")
        print("1. Place an audio file in the project directory")
        print("2. Process audio with WhisperProcessor")
        print("3. Generate transcription with model.generate()")
        print("4. Decode output with processor.batch_decode()")
        print("5. For specific language, set forced_decoder_ids or use language/task parameters")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_indic_whisper()