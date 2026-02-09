# AI Video Knowledge Editor

A specialized AI-powered video editing tool designed to automatically enrich video content with contextual knowledge cards. The system analyzes spoken content, identifies key entities, and overlays relevant visual information sourced from Wikipedia and global news publications.

## Key Features

- **Automated Transcription**: Uses OpenAI Whisper (Tiny model) for efficient speech-to-text processing.
- **Advanced Entity Recognition**: Implements GLiner (Small-v2.1 or Multi-v2.1) for high-accuracy, zero-shot entity extraction including people, organizations, and concepts.
- **Context-Aware Retrieval**: Employs an agentic search mechanism that queries Wikipedia and News sources, ranking results using semantic embeddings to ensure contextual relevance.
- **Performance Optimization**: Toggle between English-optimized and Multilingual modes to balance speed and language support.
- **Usage Analytics**: Integrated PostHog tracking for anonymous event monitoring and application performance metrics.
- **Seamless Rendering**: Automatically overlays high-quality website screenshots as knowledge cards using thum.io and FFmpeg.

## New Features

- **Cloud Analytics**: Integrated PostHog integration for anonymous usage tracking and performance monitoring.
- **Model Efficiency Modes**: Selective initialization between English-optimized and Multilingual models to optimize for hardware or language requirements.
- **Website Screenshots**: Shifted from local card generation to high-fidelity thum.io web captures for more authentic information cards.
- **Real-time Feedback**: Added an integrated terminal log and progress bar for transparent model downloads and processing status.
- **Interactive Bibliography**: Automatically exports a CSV file (`_knowledge_links.csv`) alongside every rendered video, containing timestamps and article URLs.
- **Custom URL Support**: Users can manually paste any article URL to capture a custom knowledge card.
- **Timeline Formatting**: Timeline segments are displayed in MM:SS format for improved navigation.

## Project Structure

- **py/**: Core Python implementation.
  - **gui.py**: Graphical User Interface developed with PySide6.
  - **main.py**: Application entry point with specialized compatibility patches.
  - **processor/**: Modular AI engine including speech-to-text, NLP, retrieval, and rendering engines.
  - **processor/tracker_cloud.py**: PostHog analytics implementation.
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
- **Semantic Ranking**: Sentence-Transformers (all-MiniLM-L6-v2 / L12-v2).
- **Analytics**: PostHog.
- **Video Processing**: FFmpeg (libx264).
- **Microservice Framework**: FastAPI and Uvicorn.

## Packaging & Release

### Automated Releases (Recommended)
This project is configured with GitHub Actions to automatically build standalone binaries for Windows, macOS, and Linux.
1. Push a version tag to your repository (e.g., `git tag v1.0.0 && git push origin v1.0.0`).
2. Navigate to the **Actions** tab in your GitHub repository.
3. The "Package Application" workflow will generate the binaries and create a GitHub Release.

### Local Builds
You can generate a standalone executable on your own machine using the provided scripts:

**Linux:**
```bash
cd py
bash build_linux.sh
```

**Windows:**
1. Open Command Prompt or PowerShell in the `py` directory.
2. Run `build_windows.bat`.

**Standalone Note**: The resulting executables are "one-file" bundles. Because they include high-performance libraries like Torch and FFmpeg bindings, the initial file size will be large (approx. 1.2GB - 2GB).
