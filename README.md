# AI Video Knowledge Editor

A specialized AI-powered video editing tool designed to automatically enrich video content with contextual knowledge cards. The system analyzes spoken content, identifies key entities, and overlays relevant visual information sourced from Wikipedia and global news publications.

## Key Features

- **Automated Transcription**: Uses OpenAI Whisper (Tiny model) for efficient speech-to-text processing.
- **Advanced Entity Recognition**: Implements GLiner (Small-v2.1) for high-accuracy, zero-shot entity extraction including people, organizations, and concepts.
- **Context-Aware Retrieval**: Employs an agentic search mechanism that queries Wikipedia and News sources, ranking results using semantic embeddings (all-MiniLM-L6-v2) to ensure contextual relevance.
- **Seamless Rendering**: Automatically overlays high-quality article screenshots as knowledge cards using FFmpeg at precise video timestamps.
- **Optimized Performance**: Configured with lightweight English-optimized models to ensure minimal memory footprint and faster processing times.

## Project Structure

- **py/**: Core Python implementation.
  - **gui.py**: Graphical User Interface developed with PySide6.
  - **main.py**: Application entry point with specialized compatibility patches.
  - **processor/**: Modular AI engine including speech-to-text, NLP, retrieval, and rendering engines.
  - **ml_service/**: FastAPI-based microservice for remote video analysis.
  - **setup_models.py**: Utility to pre-cache all required AI models.
- **go/**: Infrastructure for Go-based services and extensions.

## Setup and Installation

### 1. Prerequisites
- FFmpeg must be installed and available in your system path.
- Python 3.10 or higher.

### 2. Dependency Installation
Navigate to the Python directory and initialize the virtual environment:

```bash
cd py
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Model Pre-download (Recommended)
To avoid delays during the first run, download the required AI models to the local cache:

```bash
./.venv/bin/python setup_models.py
```

### 4. Running the Application
Launch the desktop editor:

```bash
bash run.sh
```

To run the ML microservice:

```bash
bash run_service.sh
```

## Technical Specifications

- **Frontend**: PySide6 (Qt for Python).
- **Audio Processing**: faster-whisper.
- **NLP Engine**: GLiNER (Zero-shot Named Entity Recognition).
- **Information Retrieval**: Wikipedia API and DuckDuckGo Search.
- **Semantic Ranking**: Sentence-Transformers (all-MiniLM-L6-v2).
- **Video Processing**: FFmpeg (libx264).
- **Microservice Framework**: FastAPI and Uvicorn.
