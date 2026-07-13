---
title: Aabha - Accessible AI Image Caption & Audio Narrator
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# Aabha: Accessible AI Image Caption & Audio Narrator

Aabha is a web application designed to empower visually impaired individuals by instantly converting uploaded images into descriptive audio narration. The application utilizes a state-of-the-art vision-language model to generate captions, translates them if necessary, and uses text-to-speech to read them aloud.

Built with a clean, fully accessible HTML5 frontend and powered by a FastAPI backend running CPU-optimized PyTorch models, Aabha is completely free to run and prioritizes user privacy (no images or audio are stored on disk).

---

## 🌟 Key Features

1. **AI Captioning**: Uses the pre-trained `Salesforce/blip-image-captioning-base` vision-language transformer model.
2. **Audio Narration**: Converts caption text into natural-sounding speech using Google Text-to-Speech (`gTTS`).
3. **Multilingual Support**: Supports English and Hindi translations (powered by the free `googletrans` API).
4. **WCAG-Compliant Frontend**: High contrast, screen-reader friendly (ARIA labels), and completely keyboard navigable (supports `Tab` focus and key triggers).
5. **In-Memory Pipeline**: All file reading, processing, translation, and TTS generation occur entirely in memory for absolute privacy.

---

## 🛠️ Tech Stack

- **Frontend**: Accessible Semantic HTML5, Vanilla CSS (Glassmorphism theme), and Vanilla JavaScript.
- **Backend API**: FastAPI (Python 3.10+).
- **Core AI Model**: Hugging Face Transformers (`BLIP`), PyTorch (CPU-only build).
- **Text-to-Speech**: `gTTS` (Google Text-to-Speech).
- **Translation**: `googletrans` (using the free `googletrans==4.0.0-rc1` release).

---

## 📂 Project Directory Structure

```text
ai-caption-generator/
├── backend/
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_pipeline.py     # Pipeline & endpoint integration tests
│   ├── caption_model.py          # Model loading and generation wrapper
│   ├── main.py                   # FastAPI server, CORS, Static files mount
│   ├── requirements.txt          # Python dependencies
│   ├── translate.py              # Google Translate API abstraction
│   └── tts.py                    # Text-To-Speech generation wrapper
├── frontend/
│   ├── index.html                # Accessible landing page structure
│   ├── script.js                 # Drag & drop, validation, fetch requests
│   └── style.css                 # Premium high-contrast styling
└── README.md                     # Documentation & HF Space configuration
```

---

## 🚀 Setup & Installation (Local Execution)

Follow these steps to run the project locally on your system.

### 1. Clone the repository
```bash
git clone https://github.com/Sakshisharan12/Aabha.git
cd Aabha/ai-caption-generator
```

### 2. Create and Activate a Virtual Environment
**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
To ensure the app runs quickly and for free, install the **CPU-only build of PyTorch** first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining dependencies:
```bash
pip install -r backend/requirements.txt
```

### 4. Run the Application
Start the FastAPI server:
```bash
cd backend
python main.py
```
Or run directly via Uvicorn:
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Once running, open your web browser and navigate to **`http://127.0.0.1:8000`**. The FastAPI server automatically mounts and serves the static frontend files.

---

## 🧪 Running Automated Tests

A comprehensive test suite is included in `backend/tests/` to verify captioning, translation, text-to-speech, and image format conversion.

Run the test suite using `pytest`:
```bash
cd backend
python -m pytest tests/ -v
```

---

## ☁️ Hugging Face Spaces Deployment

Aabha can be deployed directly to Hugging Face Spaces using Docker.

### 1. Create a `Dockerfile`
Create a `Dockerfile` at the root of `ai-caption-generator` folder:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to avoid heavy GPU installs
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

EXPOSE 8000

WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Push to Hugging Face Spaces
Create a new Space on Hugging Face with **Docker** SDK, then run:
```bash
git init
git remote add origin https://huggingface.co/spaces/<your-username>/aabha
git add .
git commit -m "Deploy to HF Spaces with Docker"
git push -u origin main -f
```

---

## ⚠️ Known Limitations

- **Text-Heavy Images (OCR)**: The BLIP model is a general-purpose caption generator and does not perform OCR. It will not transcribe documents or dense text accurately.
- **Real-Time Video**: Aabha is optimized for static image uploads only. Real-time video stream captioning is out of scope for this version.
- **Hallucinations**: Like all transformer-based vision models, BLIP may occasionally hallucinate or misidentify objects in low-light, blurry, or complex overlapping scenes.
