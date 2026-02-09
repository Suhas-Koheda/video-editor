@echo off
echo Setting up build environment...
pip install pyinstaller -r requirements.txt

echo Starting PyInstaller build (this may take several minutes)...
pyinstaller --noconfirm --onefile --windowed ^
    --name "KnowledgeEditor" ^
    --collect-all torch ^
    --collect-all gliner ^
    --collect-all faster_whisper ^
    --hidden-import "av.subtitles" ^
    --hidden-import "av.subtitles.stream" ^
    main.py

echo ------------------------------------------------
echo Build complete! Check the 'dist' folder.
echo ------------------------------------------------
pause
