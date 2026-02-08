# AI Video Knowledge Editor

A professional-grade AI video editor that automatically enriches your videos with knowledge cards from Wikipedia and News sources. It uses state-of-the-art Multilingual NLP and Speech-to-Text to understand your content and overlay relevant information.

## 🚀 Features
- **Speech to Text**: Multi-language transcription using OpenAI Whisper.
- **Entity Extraction**: Zero-shot Named Entity Recognition (NER) using GLiNER.
- **Agentic Search**: Intelligent search across Wikipedia and global News sources.
- **Semantic Ranking**: Results are ranked by relevance to your video's context.
- **Professional Overlays**: Automated FFmpeg-based knowledge card overlays.

## 📁 Project Structure
- **/py**: Python implementation (GUI, AI Processing).
  - `gui.py`: The professional editor interface.
  - `processor/`: The AI engine modules.
  - `run.sh`: Startup script.
- **/go**: Placeholder for Go-based implementations/services.

## 🛠️ Setup & Usage

### 1. Requirements
Ensure you have `ffmpeg` installed on your system.

### 2. Installation
Navigate to the `py` directory and install dependencies:
```bash
cd py
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Preheat Models (Optional but Recommended)
Download the AI models beforehand to ensure smooth editing:
```bash
./setup_models.py
```

### 4. Run the Editor
```bash
./run.sh
```

## 🧠 Technology Stack
- **GUI**: PySide6 (Qt)
- **AI Models**: 
  - `faster-whisper` (Speech)
  - `gliner` (NER)
  - `sentence-transformers` (Ranking)
- **Backend**: Python 3.10+
- **Video Engine**: FFmpeg
