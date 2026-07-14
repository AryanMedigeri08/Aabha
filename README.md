---
title: Aabha - Accessible AI Image Caption & Audio Narrator
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# Aabha: Accessible Next.js AI Image Caption & Audio Narrator

Aabha is a web application designed to empower visually impaired individuals by instantly converting uploaded images into descriptive audio narration. The application utilizes a state-of-the-art vision-language model to generate captions, translates them if necessary, and uses text-to-speech to read them aloud.

Aabha features a modern, fully accessible React **Next.js (App Router)** frontend and a high-performance **FastAPI** backend running CPU-optimized PyTorch models.

---

## 🌟 Key Features

1. **AI Captioning**: Uses the pre-trained `Salesforce/blip-image-captioning-base` vision-language transformer model.
2. **Audio Narration**: Converts caption text into natural-sounding speech using Google Text-to-Speech (`gTTS`).
3. **Multilingual Support**: Supports English and Hindi translations (powered by the free `googletrans` API).
4. **WCAG-Compliant Frontend**: High contrast, screen-reader friendly (ARIA labels), and completely keyboard navigable (supports `Tab` focus and key triggers).
5. **In-Memory Pipeline**: All file reading, processing, translation, and TTS generation occur entirely in memory for absolute privacy.

---

## 🛠️ Tech Stack

- **Frontend**: Next.js (App Router), React, Vanilla CSS (Glassmorphism theme), Lucide Icons.
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
│   ├── main.py                   # FastAPI server & CORS setup
│   ├── requirements.txt          # Python dependencies
│   ├── translate.py              # Google Translate API abstraction
│   └── tts.py                    # Text-To-Speech generation wrapper
├── frontend/
│   ├── src/
│   │   └── app/
│   │       ├── layout.js         # Next.js app wrapper & fonts
│   │       ├── page.js           # Core React Single Page App UI
│   │       └── globals.css       # Premium styles and resets
│   ├── package.json              # Node.js dependencies
│   ├── jsconfig.json             # JS path configs
│   └── next.config.mjs           # Next.js config (Static exports & Dev proxies)
├── Dockerfile                    # Multi-stage build setup
└── README.md                     # Documentation & HF Space configuration
```

---

## 🚀 Setup & Installation (Local Execution)

Running Aabha locally in development mode requires running both the FastAPI backend and the Next.js frontend in parallel.

### 1. Clone the repository
```bash
git clone https://github.com/Sakshisharan12/Aabha.git
cd Aabha/ai-caption-generator
```

### 2. Configure and Run the Backend API

**Create and Activate a Virtual Environment:**
*On Windows:*
```bash
python -m venv venv
venv\Scripts\activate
```
*On macOS/Linux:*
```bash
python -m venv venv
source venv/bin/activate
```

**Install Backend Dependencies:**
```bash
# Install PyTorch CPU-only build first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r backend/requirements.txt
```

**Start Backend Server:**
```bash
cd backend
python main.py
```
The backend API is now running on **`http://localhost:8000`**.

---

### 3. Configure and Run the Next.js Frontend

**Install Node Dependencies:**
Open a new terminal session, navigate to the `frontend/` folder, and run:
```bash
cd ai-caption-generator/frontend
npm install
```

**Start Next.js Development Server:**
```bash
npm run dev
```
The frontend application is now running on **`http://localhost:3000`**. 
Next.js automatically proxies API calls made to `/api/*` to the FastAPI backend on port `8000`.

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

Aabha can be deployed directly to Hugging Face Spaces using the provided multi-stage Docker build.

Next.js is built as a **static HTML export** during the first Docker stage, and served directly by the FastAPI backend in the second stage. This removes the need for a separate Node.js server in production!

### Push to Hugging Face Spaces
Create a new Space on Hugging Face with **Docker** SDK, then run:
```bash
git init
git remote add origin https://huggingface.co/spaces/<your-username>/aabha
git add .
git commit -m "Deploy to HF Spaces with Multi-Stage Docker"
git push -u origin main -f
```

---

## ⚠️ Known Limitations

- **Text-Heavy Images (OCR)**: The BLIP model is a general-purpose caption generator and does not perform OCR. It will not transcribe documents or dense text accurately.
- **Real-Time Video**: Aabha is optimized for static image uploads only. Real-time video stream captioning is out of scope for this version.
- **Hallucinations**: Like all transformer-based vision models, BLIP may occasionally hallucinate or misidentify objects in low-light, blurry, or complex overlapping scenes.
