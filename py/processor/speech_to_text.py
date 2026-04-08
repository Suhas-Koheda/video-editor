import os
import torch
import requests
from processor.config import get_whisper_model, get_stt_engine, get_sarvam_config

CACHE_DIR = os.path.join(os.getcwd(), ".model_cache")

_whisper_model = None

def transcribe_audio_with_timestamps(audio_path, video_path=None):
    """
    Transcribe audio with timestamps using either Whisper or Sarvam AI.
    """
    engine = get_stt_engine()
    print(f"[DEBUG] Selected STT Engine: {engine}")
    
    if engine == "sarvam":
        return _transcribe_sarvam(audio_path, video_path)
    else:
        return _transcribe_whisper(audio_path)

def _transcribe_whisper(audio_path, video_path=None):
    from faster_whisper import WhisperModel
    global _whisper_model
    print(f"[DEBUG] Starting Whisper transcription for: {audio_path}")
    if _whisper_model is None:
        model_name = get_whisper_model()
        print(f"[DEBUG] Loading Whisper model: {model_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _whisper_model = WhisperModel(
            model_name,
            device=device,
            compute_type="int8" if device == "cpu" else "float16",
            download_root=os.path.join(CACHE_DIR, "whisper")
        )

    segments, info = _whisper_model.transcribe(audio_path, beam_size=5)

    results = []
    for segment in segments:
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    return results, info.language

def _transcribe_sarvam(audio_path, video_path=None):
    """
    Transcribe using Sarvam AI (API).
    Handles files longer than 30s by chunking locally.
    """
    print(f"[DEBUG] Starting Sarvam AI transcription for: {audio_path}")
    config = get_sarvam_config()
    api_key = config["api_key"]

    if not api_key:
        print("[DEBUG] ERROR: No Sarvam API key found. Falling back to Whisper.")
        return _transcribe_whisper(audio_path)

    # Chunk the audio into 29s pieces to stay safely under the 30s limit
    os.makedirs("temp/chunks", exist_ok=True)
    import subprocess
    import glob
    
    # Clear old chunks
    for f in glob.glob("temp/chunks/chunk_*.wav"):
        try: os.remove(f)
        except: pass
        
    print(f"[DEBUG] Splitting audio into 29s chunks...")
    subprocess.run([
        "ffmpeg", "-i", audio_path,
        "-f", "segment", "-segment_time", "29",
        "-c", "copy", "temp/chunks/chunk_%03d.wav", "-y"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    chunk_files = sorted(glob.glob("temp/chunks/chunk_*.wav"))
    if not chunk_files:
        print("[DEBUG] ERROR: Failed to chunk audio.")
        return _transcribe_whisper(audio_path)

    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": api_key}
    
    all_results = []
    total_offset = 0.0
    
    for i, chunk_path in enumerate(chunk_files):
        print(f"[DEBUG] Processing chunk {i+1}/{len(chunk_files)}: {chunk_path}")
        try:
            with open(chunk_path, "rb") as f:
                files = {"file": (os.path.basename(chunk_path), f, "audio/wav")}
                data = {"model": "saaras:v3"}
                response = requests.post(url, headers=headers, files=files, data=data)
            
            if response.status_code == 200:
                transcript = response.json().get("transcript", "")
                words = transcript.split()
                # Simple segmentation of the transcript for this chunk
                words_per_seg = 12
                chunk_duration = 29.0 # Assume approx 29s except maybe for last
                
                for j in range(0, len(words), words_per_seg):
                    seg_text = " ".join(words[j:j+words_per_seg])
                    # Estimated timestamps within the chunk
                    rel_start = (j / max(1, len(words))) * chunk_duration
                    rel_end = ((j + words_per_seg) / max(1, len(words))) * chunk_duration
                    
                    all_results.append({
                        "start": total_offset + rel_start,
                        "end": total_offset + rel_end,
                        "text": seg_text
                    })
                total_offset += 29.0
            else:
                print(f"[DEBUG] Sarvam Chunk {i} Failed ({response.status_code}): {response.text}")
        except Exception as e:
            print(f"[DEBUG] Error processing chunk {i}: {e}")

    if not all_results:
        print("[DEBUG] No transcripts received from Sarvam. Falling back.")
        return _transcribe_whisper(audio_path)

    # Heuristic for language detection
    search_path = (video_path or audio_path).lower()
    if "telugu" in search_path or "te" in search_path:
        lang = "te"
    elif "hindi" in search_path or "hi" in search_path:
        lang = "hi"
    elif "marathi" in search_path or "mr" in search_path:
        lang = "mr"
    elif "tamil" in search_path or "ta" in search_path:
        lang = "ta"
    else:
        lang = "hi" # Default to hindi for indian context

    print(f"[DEBUG] Sarvam complete. Total segments: {len(all_results)}. Heuristic Language: {lang}")
    return all_results, lang

def unload_whisper_model():
    """
    Clears the STT models from memory.
    """
    global _whisper_model
    import gc
    if _whisper_model is not None:
        del _whisper_model
        _whisper_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
