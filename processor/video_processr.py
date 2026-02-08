import os
import subprocess

def extract_audio(video_path):
    os.makedirs("temp", exist_ok=True)
    audio_path = "temp/audio_for_transcription.wav"

    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-ar", "16000", "-ac", "1",
        audio_path, "-y"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return audio_path

# Note: The high-level process is now managed by gui.py via the AnalysisWorker
# to allow for the interactive editor experience requested.
