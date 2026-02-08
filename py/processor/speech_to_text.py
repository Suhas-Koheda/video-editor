from faster_whisper import WhisperModel
import os

CACHE_DIR = os.path.join(os.getcwd(), ".model_cache")

_model = None

def transcribe_audio_with_timestamps(audio_path):
    """
    Transcribe audio with timestamps.
    """
    global _model
    if _model is None:
        print("Loading Whisper 'base' model for Speech-to-Text...")
        _model = WhisperModel(
            "tiny", 
            device="cpu", 
            compute_type="int8", 
            download_root=os.path.join(CACHE_DIR, "whisper")
        )
        print("✓ Whisper loaded.")
    
    segments, info = _model.transcribe(audio_path, beam_size=5)
    
    results = []
    for segment in segments:
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
    
    return results, info.language

def unload_whisper_model():
    """
    Clears the Whisper model from memory.
    """
    global _model
    if _model is not None:
        import gc
        del _model
        _model = None
        gc.collect()
        print("Whisper model unloaded successfully.")