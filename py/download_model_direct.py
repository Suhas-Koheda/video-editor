#!/usr/bin/env python3
"""
Direct model download to ensure files are properly cached
"""
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from huggingface_hub import snapshot_download

def download_telugu_whisper():
    model_name = "steja/whisper-small-telugu"
    cache_dir = "/home/ssp/ML/proejct/py/.model_cache"
    
    print(f"Downloading {model_name}...")
    
    try:
        # Download using snapshot_download to get all files
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_files_only=False
        )
        print(f"Model downloaded to: {model_path}")
        
        # Verify we can load the model
        print("Verifying model loading...")
        processor = WhisperProcessor.from_pretrained(model_path)
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Processor type: {type(processor)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    download_telugu_whisper()