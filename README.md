# AI Video Knowledge Editor

**Transform standard video content into interactive, knowledge-rich experiences.**

The AI Video Knowledge Editor is a state-of-the-art tool that automatically enriches video content with contextual "Knowledge Cards." By combining high-speed transcription, zero-shot entity recognition, and real-time web retrieval, it overlays relevant information from Wikipedia and global news sources directly onto your video.

---

## Core Pipeline: How it Works

The application follows a sophisticated multi-stage processing pipeline to ensure contextual accuracy and visual quality:

1.  **Speech-to-Text**: Extracts audio and transcribes it using `faster-whisper` (Tiny/Base models).
2.  **Multilingual Intelligence**: Detects the language and provides high-fidelity translation (for Indic and global languages) using Sarvam AI (API) or Facebook's NLLB-200 (Local).
3.  **Global Entity Extraction**: Uses **GLiNER** for zero-shot Named Entity Recognition (NER), identifying people, concepts, and events.
4.  **Contextual Ranking**: Analyzes the entire video's transcript to rank entities by importance using a "Sliding Window" context and global frequency statistics.
5.  **Agentic Retrieval**: Queries Wikipedia and DuckDuckGo for the most relevant articles and summaries.
6.  **Visual Rendering**: Captures high-fidelity screenshots using **Playwright (Chromium)** and overlays them as stylized cards via **FFmpeg**.

---

## Key Features

### Advanced Multilingual Support
Native optimization for English and **Indic Languages** (Hindi, Telugu, Tamil, Kannada, etc.).
- **Sarvam AI Integration**: Blazing fast API-based translation for Indian dialects.
- **Local NLLB Fallback**: Full privacy with Facebook's "No Language Left Behind" model running on your hardware.

### Intelligent Entity Ranking
Unlike simple NER, this system understands the "narrative" of the video:
- **Global Context Awareness**: Entities are prioritized based on how often they appear across the *entire* video.
- **Cross-Segment Context**: Analyzing neighboring segments to resolve ambiguity.
- **Dynamic Scoring**: Entities are weighted by frequency, phrase complexity, and appearance timing.

### High-Fidelity Knowledge Cards
- **Playwright Rendering**: Real-time browser-based screenshots for authentic information displays.
- **Custom URL Overrides**: Manually paste any URL to force a specific knowledge card for a segment.
- **Interactive Bibliography**: Every video export includes a `_knowledge_links.csv` file with timestamps and source URLs.

### Professional Analytics & Monitoring
- **PostHog Integration**: Privacy-conscious event tracking for performance optimization.
- **Real-time Logs**: Integrated progress tracking for model downloads and rendering stages.

---

## Technical Stack

| Category | Technology |
| :--- | :--- |
| **Frontend** | PySide6 (Qt for Python) |
| **Audio AI** | faster-whisper |
| **NLP Engine** | GLiNER (Small-v2.1 / Multi-v2.1) |
| **Translation** | Sarvam AI API / NLLB-200 |
| **Search** | Wikipedia API, DuckDuckGo (DDGS) |
| **Rendering** | FFmpeg, Playwright (Chromium) |
| **Backend** | FastAPI (Microservice mode) |

---

## Setup & Installation

### 1. Prerequisites
- **Python 3.10+**
- **FFmpeg**: Must be installed and added to your System PATH.
- **Node.js** (Optional, for Playwright dependencies)

### 2. Dependency Installation
```bash
cd py
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

### 3. Model Pre-download (Recommended)
Pre-cache the AI models (approx. 2GB) to ensure smooth first-time execution:
```bash
python setup_models.py
```

---

## Usage

### Desktop GUI
Launch the main application for a visual editing experience:
```bash
bash run.sh  # Windows: run_windows.bat
```

### Headless ML Service
Run the analysis engine as a FastAPI microservice:
```bash
bash run_service.sh
```

---

## Project Structure

- **`py/`**: Core Python implementation.
  - **`gui.py`**: Rich desktop interface.
  - **`processor/`**: The modular AI engine (Speech, Translation, NLP, Retrieval, Rendering).
  - **`ml_service/`**: FastAPI implementation for remote processing.
- **`go/`**: Infrastructure extensions for high-performance services.
- **`.github/`**: Automated CI/CD pipelines for building standalone binaries.

---

## Roadmap & Development

- [x] Indic Language Support (Beta)
- [x] Global Entity Ranking logic
- [x] Local Playwright rendering
- [ ] Real-time "Knowledge Stream" preview
- [ ] Support for custom CSS themes in Knowledge Cards

> [!TIP]
> To optimize for speed, use the **English-only** mode. For deep contextual analysis of regional content, enable the **Multilingual** model in Settings.

---
*Developed for the next generation of educational and news content.*
