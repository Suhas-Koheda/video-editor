import os
import subprocess

def extract_audio(video_path):
    """
    Extract audio from video.
    """
    print(f"[DEBUG] Extracting audio from: {video_path}")
    os.makedirs("temp", exist_ok=True)
    audio_path = "temp/audio_for_transcription.wav"

    try:
        result = subprocess.run([
            "ffmpeg", "-i", video_path,
            "-ar", "16000", "-ac", "1",
            audio_path, "-y"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[DEBUG] FFmpeg ERROR: {result.stderr}")
        
        if os.path.exists(audio_path):
            size = os.path.getsize(audio_path)
            print(f"[DEBUG] Audio extracted to {audio_path} ({size} bytes)")
        else:
            print(f"[DEBUG] ERROR: Audio file was not created by FFmpeg")
            
    except Exception as e:
        print(f"[DEBUG] ERROR during audio extraction: {e}")

    return audio_path
