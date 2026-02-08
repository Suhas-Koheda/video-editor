from faster_whisper import WhisperModel

_model = None

def transcribe_audio_with_timestamps(audio_path):
    global _model
    if _model is None:
        _model = WhisperModel("base", device="cpu", compute_type="int8")
    
    segments, info = _model.transcribe(audio_path, beam_size=5)
    
    results = []
    for segment in segments:
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
    
    return results