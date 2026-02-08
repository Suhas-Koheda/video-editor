from faster_whisper import WhisperModel

_model = None

def transcribe_audio_with_timestamps(audio_path):
    global _model
    if _model is None:
        # 'base' model is multilingual and very efficient
        _model = WhisperModel("base", device="cpu", compute_type="int8")
    
    # transcribe returns (segments, info)
    # info contains the detected language: info.language
    segments, info = _model.transcribe(audio_path, beam_size=5)
    
    results = []
    for segment in segments:
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
    
    return results, info.language