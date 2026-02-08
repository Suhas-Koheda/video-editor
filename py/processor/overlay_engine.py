import subprocess
import os

def render_with_screenshots(video_path, render_plan):
    """
    Renders the final video by overlaying Wikipedia screenshots at specified timestamps.
    render_plan is a list of segments: [{'start', 'end', 'screenshot_path'}, ...]
    """
    output_path = "output/final_project.mp4"
    os.makedirs("output", exist_ok=True)

    inputs = ["-i", video_path]
    filter_parts = []
    prev_label = "[0:v]"
    
    for i, seg in enumerate(render_plan):
        inputs.extend(["-i", seg['screenshot_path']])
        
        label = f"[v{i+1}]"
        screenshot_idx = i + 1
        
        part = f"[{screenshot_idx}:v]scale=400:-1[img{i}]; {prev_label}[img{i}]overlay=x=40:y=40:enable='between(t,{seg['start']},{seg['end']})'"
        
        if i == len(render_plan) - 1:
            filter_parts.append(part)
        else:
            filter_parts.append(part + f"{label}; ")
            prev_label = label

    filter_complex = "".join(filter_parts)

    command = [
        "ffmpeg",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "0:a?",
        "-codec:v", "libx264",
        "-preset", "veryfast",
        "-y",
        output_path
    ]

    print(f"Executing Rendering: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"FFmpeg Rendering Failed: {result.stderr}")
        
    return output_path