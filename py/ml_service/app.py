import sys
import types
import av

if not hasattr(av, 'subtitles'):
    sub_mod = types.ModuleType("subtitles")
    av.subtitles = sub_mod
    sys.modules["av.subtitles"] = sub_mod

    stream_mod = types.ModuleType("stream")

    SubtitleStream = type("SubtitleStream", (), {})
    stream_mod.SubtitleStream = SubtitleStream
    sub_mod.SubtitleStream = SubtitleStream

    sub_mod.stream = stream_mod
    sys.modules["av.subtitles.stream"] = stream_mod

from fastapi import FastAPI, UploadFile, File
import shutil
import os

from processor.video_processor import extract_audio
from processor.speech_to_text import transcribe_audio_with_timestamps
from processor.nlp_engine import get_entities_and_nouns

app = FastAPI()

TEMP_DIR = "temp_service"
os.makedirs(TEMP_DIR, exist_ok=True)


@app.get("/")
def health():
    return {"status": "ML Service Running"}


@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):

    video_path = os.path.join(TEMP_DIR, file.filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    audio_path = extract_audio(video_path)

    segments, language = transcribe_audio_with_timestamps(audio_path)

    for seg in segments:
        seg["entities"] = get_entities_and_nouns(seg["text"])
        seg["language"] = language

    # --- NEW PIPELINE STEPS ---
    from processor.nlp_engine import build_global_entity_stats, compute_global_scores, get_sliding_context, rank_entities_for_segment
    
    global_stats = build_global_entity_stats(segments)
    global_stats = compute_global_scores(global_stats)

    for i, seg in enumerate(segments):
        context_text = get_sliding_context(segments, i)
        local_entities = seg.get('entities', [])
        
        # Rank entities using global importance and sliding window context
        final_ranked = rank_entities_for_segment(seg['text'], local_entities, global_stats, context_text)
        seg['final_entities'] = final_ranked
    # ---------------------------

    return {
        "language": language,
        "segments": segments
    }
