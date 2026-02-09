#!/bin/bash
# Local Build Script for Linux

echo "Setting up build environment..."
./.venv/bin/pip install pyinstaller

echo "Starting PyInstaller build (this may take several minutes)..."
./.venv/bin/pyinstaller --noconfirm --onefile --windowed \
    --name "KnowledgeEditor" \
    --collect-all torch \
    --collect-all gliner \
    --collect-all faster_whisper \
    --hidden-import "av.subtitles" \
    --hidden-import "av.subtitles.stream" \
    main.py

echo "------------------------------------------------"
echo "Build complete! Check the 'dist' folder."
echo "------------------------------------------------"
