# System Architecture
## AI Caption Generator — Accessible Image-to-Speech App

---

## 1. Architecture Overview

The system is a single-service web application (no separate backend/frontend split) built with Streamlit, which handles UI, orchestration, and inference in one Python process. This keeps the architecture simple enough for an 8-week beginner project while still demonstrating a real multimodal transformer pipeline.

```
┌──────────────────────────────────────────────────────────────────┐
│                         Streamlit App (single process)           │
│                                                                    │
│  ┌───────────────┐     ┌──────────────────┐    ┌───────────────┐ │
│  │  Upload / UI  │────▶│  BLIP Captioning  │───▶│ Caption (EN)  │ │
│  │  (drag-drop)  │     │  Model (PyTorch)  │    │   text        │ │
│  └───────────────┘     └──────────────────┘    └───────┬───────┘ │
│                                                          │         │
│                                     ┌────────────────────┼──────┐  │
│                                     ▼                    ▼      │  │
│                          ┌───────────────────┐  ┌───────────────┴─┐│
│                          │ Google Translate  │  │      gTTS       ││
│                          │  API (EN → HI)    │  │ (text → speech) ││
│                          └─────────┬─────────┘  └────────┬────────┘│
│                                    ▼                     ▼         │
│                          ┌───────────────────┐  ┌───────────────┐  │
│                          │  Caption (HI) text│  │  Audio file   │  │
│                          └───────────────────┘  │  (mp3, in-mem)│  │
│                                                  └───────┬───────┘  │
│                                                          ▼          │
│                                              ┌───────────────────┐  │
│                                              │ Browser audio      │  │
│                                              │ player + download  │  │
│                                              └───────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                             Deployed on Hugging Face Spaces
```

## 2. Components

### 2.1 Frontend / UI Layer
- **Framework:** Streamlit
- **Responsibilities:**
  - Render drag-and-drop / file-picker upload widget
  - Show image preview
  - Show generated caption (English + optional Hindi)
  - Render `st.audio()` widget for playback and download
  - Language toggle (English / Hindi)
  - Accessibility: ARIA-friendly labels, high-contrast theme, keyboard-navigable widgets

### 2.2 Captioning Engine (Core AI)
- **Model:** `Salesforce/blip-image-captioning-base` (HuggingFace Transformers)
- **Type:** Vision-Language transformer (ViT-based image encoder + text decoder, cross-attention between modalities)
- **Responsibilities:**
  - Preprocess uploaded image (resize, normalize) via `BlipProcessor`
  - Run forward pass through `BlipForConditionalGeneration`
  - Decode output tokens into an English caption string
- **Execution:** CPU inference (no GPU required); model loaded once and cached in memory across requests using Streamlit's `@st.cache_resource`

### 2.3 Translation Layer
- **Service:** Google Translate API (or `googletrans` as a free fallback)
- **Responsibilities:** Translate the English caption into Hindi text when the user selects the Hindi option

### 2.4 Text-to-Speech Layer
- **Library:** gTTS (Google Text-to-Speech)
- **Responsibilities:**
  - Convert caption text (English or Hindi, `lang='en'` / `lang='hi'`) into an MP3 audio stream
  - Return audio as an in-memory byte stream (no disk persistence, for privacy)

### 2.5 Deployment
- **Platform:** Hugging Face Spaces (Streamlit SDK)
- **Compute:** Free-tier CPU instance
- **Config:** `requirements.txt` pinning transformers, torch (CPU build), streamlit, gTTS, googletrans/Google Translate client, Pillow

## 3. Data Flow (Step by Step)

1. User uploads/drops an image → stored transiently in memory (PIL `Image` object), never written to disk.
2. Image passed to `BlipProcessor` → tensor input for the model.
3. `BlipForConditionalGeneration.generate()` produces token IDs → decoded to an English caption string.
4. Caption displayed in UI immediately.
5. If Hindi selected: caption text sent to Translate API → Hindi string returned and displayed.
6. Caption (EN or HI) passed to gTTS → MP3 bytes generated in memory.
7. MP3 bytes rendered via `st.audio()` for playback; download button offered.
8. Session ends → all in-memory image/audio data discarded (no persistence layer).

## 4. Technology Stack Summary

| Layer | Technology |
|-------|-----------|
| AI Model | Salesforce/blip-image-captioning-base (HuggingFace Transformers, PyTorch) |
| Image handling | PIL / Pillow |
| Text-to-speech | gTTS |
| Translation | Google Translate API (or googletrans) |
| Frontend/UI | Streamlit |
| Deployment | Hugging Face Spaces |
| Language | Python 3.10+ |

## 5. Why This Architecture

- **Single-process simplicity:** Appropriate for an 8-week, beginner-friendly project — avoids the complexity of a separate FastAPI backend + React frontend (used in more advanced projects like the Attention Head Explorer).
- **No training required:** BLIP is used as a pretrained model, keeping the "AI complexity" high in demo impact but low in engineering overhead.
- **Free-tier friendly:** Every component (BLIP on CPU, gTTS, HF Spaces) is free, so the app is fully reproducible with zero infrastructure cost.
- **Privacy by design:** No database or file storage means no user image data persists after the session, which matters for an accessibility-focused, personal-use tool.

## 6. Scalability & Future Extensions (Not in Scope for v1)

- Swap Streamlit for a FastAPI backend + dedicated frontend if concurrent multi-user load becomes a concern.
- Add caching of repeated image captions (hash-based) to reduce redundant inference.
- Support additional languages beyond Hindi.
- Add batch/video-frame captioning for near-real-time description.
