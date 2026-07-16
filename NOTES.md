# Aabha — Project Deep Dive (Personal Notes)

> **Private document** — gitignored, not part of the repository.

---

## 1. Project Overview

**Aabha** is an accessible web application that converts uploaded images into descriptive audio narrations, empowering visually impaired users to perceive visual content through sound. It also supports interactive visual question answering (VQA) — users can ask natural language questions about uploaded images.

### Core User Flow
```
User uploads image
    → BLIP generates English caption
    → (Optional) Google Translate converts to Hindi
    → Edge-TTS synthesises speech audio
    → Frontend plays audio automatically + shows text
    → User can ask follow-up questions via VQA chat
```

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js 14)                     │
│                        Port 3000 (dev)                           │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  page.js — Single-page React app                        │    │
│  │  • Image upload (drag & drop + file picker)             │    │
│  │  • Language selector (EN / HI)                          │    │
│  │  • Caption display + audio player with speed control    │    │
│  │  • VQA chat interface                                   │    │
│  │  • Web Audio API accessibility sound cues               │    │
│  │  • Keyboard shortcuts (U/R/L/S)                         │    │
│  └──────────────────────────────────────────────────────────┘    │
│             │  API calls proxied via next.config.mjs rewrites    │
└─────────────┼────────────────────────────────────────────────────┘
              │  POST /api/caption   (multipart: image + lang)
              │  POST /api/chat      (multipart: image + question + lang)
              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       BACKEND (FastAPI + Uvicorn)                 │
│                       Port 8000                                  │
│                                                                  │
│  main.py ─────────────────────────────────────────────────────── │
│  │  • App lifespan: loads BLIP model once at startup             │
│  │  • POST /api/caption → validate → caption → translate → TTS  │
│  │  • POST /api/chat    → validate → VQA → translate → TTS      │
│  │  • GET  /api/health  → health check                          │
│  │  • Serves static frontend build (production)                  │
│  │                                                               │
│  ├─ caption_model.py                                             │
│  │  • load_model() — loads BlipProcessor + BlipForCondGen        │
│  │  • generate_caption(image) — image → English text             │
│  │  • generate_answer(image, question) — VQA                    │
│  │                                                               │
│  ├─ tts.py                                                       │
│  │  • caption_to_audio(text, lang) — async edge-tts streaming   │
│  │  • Voices: en-US-AriaNeural, hi-IN-SwaraNeural               │
│  │                                                               │
│  ├─ translate.py                                                 │
│  │  • translate_caption(text, target_lang) — googletrans wrapper │
│  │  • Thin abstraction for easy swap to Google Cloud API         │
│  │                                                               │
│  └─ requirements.txt                                             │
└──────────────────────────────────────────────────────────────────┘
```

### Deployment Architecture (Docker)
The `Dockerfile` uses a **multi-stage build**:
1. **Stage 1 (node:18-alpine):** Builds the Next.js frontend into a static export (`out/`)
2. **Stage 2 (python:3.10-slim):** Installs PyTorch CPU + backend deps, copies backend source + built frontend. FastAPI serves the static frontend from `frontend/out/` in production.

---

## 3. How the Transformer (BLIP) is Used

### Model: `Salesforce/blip-image-captioning-base`

**BLIP** (Bootstrapping Language-Image Pre-training) is a Vision-Language Transformer model. It was published by Salesforce Research in 2022.

### Architecture Breakdown

```
                    Image                    Text (optional prompt)
                      │                              │
                      ▼                              ▼
            ┌─────────────────┐            ┌─────────────────┐
            │   Vision        │            │   Text           │
            │   Transformer   │            │   Transformer    │
            │   (ViT)         │            │   (BERT-based)   │
            │                 │            │                  │
            │  Patch embed    │            │  Token embed     │
            │  + Self-Attn    │            │  + Self-Attn     │
            │  + FFN layers   │            │  + Cross-Attn    │
            └────────┬────────┘            └────────┬─────────┘
                     │                              │
                     │    Cross-Attention            │
                     └──────────┬───────────────────┘
                                │
                                ▼
                        ┌──────────────┐
                        │  LM Head     │
                        │  (Decoder)   │
                        └──────┬───────┘
                               │
                               ▼
                        Generated Text
                  (caption or VQA answer)
```

### Key Components

| Component | What it does | Details |
|-----------|-------------|---------|
| **Vision Transformer (ViT)** | Encodes the image into a sequence of patch embeddings | Input image → 224×224 → split into 16×16 patches → linear projection → position embeddings → 12 self-attention layers |
| **BlipProcessor** | Preprocesses both image and text | Resizes/normalizes image to tensor, tokenizes text prompts |
| **Text Decoder (BERT-based)** | Generates caption text autoregressively | Uses cross-attention to attend to ViT visual features while generating tokens |
| **LM Head** | Predicts next token probability | Linear layer on top of decoder hidden states → vocabulary logits |

### How Captioning Works (generate_caption)
```python
# 1. Processor converts PIL Image → normalized tensor
inputs = _processor(image, return_tensors="pt")
# inputs contains: pixel_values tensor [1, 3, 224, 224]

# 2. Model generates token IDs autoregressively
output_ids = _model.generate(**inputs, max_new_tokens=40)
# ViT encodes image → decoder generates text tokens using cross-attention

# 3. Decode token IDs → human-readable string
caption = _processor.decode(output_ids[0], skip_special_tokens=True)
```

### How VQA Works (generate_answer)
```python
# 1. Format question as a prompt
prompt = f"Question: {question} Answer:"

# 2. Processor handles BOTH image and text
inputs = _processor(image, text=prompt, return_tensors="pt")
# inputs contains: pixel_values + input_ids + attention_mask

# 3. Model generates answer conditioned on image + question
output_ids = _model.generate(**inputs, max_new_tokens=40)
# The text decoder uses cross-attention to the visual features
# AND self-attention to the question context to produce the answer
```

### Model Specifications
| Spec | Value |
|------|-------|
| Parameters | ~224M |
| Disk size | ~990 MB (safetensors) |
| Image input | 224 × 224 pixels |
| ViT variant | ViT-B/16 (12 layers, 768 dim, 12 heads) |
| Text decoder | 12 layers, 768 dim |
| Vocabulary | 30,524 tokens (BERT tokenizer) |
| Max generation | 40 new tokens (configured) |
| Precision | FP32 (no quantization) |
| Device | CPU only |

---

## 4. Performance Characteristics

### Inference Speed (CPU, approximate)

| Operation | Time | Notes |
|-----------|------|-------|
| Model loading (first run) | ~30–60s | Downloads ~990MB model, caches locally |
| Model loading (cached) | ~5–10s | Loads safetensors from disk into RAM |
| Caption generation | ~3–8s | Depends on CPU; i7 ~3s, i5 ~6s |
| VQA answer generation | ~3–8s | Similar to captioning |
| Translation (googletrans) | ~0.5–1s | Network round-trip to Google servers |
| TTS (edge-tts) | ~1–3s | Network streaming from Microsoft Edge servers |
| **Total end-to-end** | **~5–12s** | From upload to audio playback |

### Memory Usage

| Component | RAM |
|-----------|-----|
| BLIP model (FP32) | ~1.0 GB |
| PyTorch runtime overhead | ~200 MB |
| Python + FastAPI + deps | ~150 MB |
| Per-request processing | ~50–100 MB (image tensors, temporary) |
| **Total steady-state** | **~1.4–1.5 GB** |

### Bottlenecks
1. **CPU inference is the main bottleneck.** BLIP's ViT encoder runs 12 transformer layers on every image — this is compute-heavy without GPU.
2. **FP32 precision.** No quantization is applied, so the model uses full 32-bit floating point.
3. **Sequential processing.** Each request blocks the model — there's no batching or request queuing.
4. **Network-dependent TTS/Translation.** Both `edge-tts` and `googletrans` make external API calls.

---

## 5. Technology Stack Summary

| Layer | Technology | Version | Role |
|-------|-----------|---------|------|
| Frontend framework | Next.js | 14.2.x | React SSR/SSG framework |
| Frontend runtime | React | 18.3.x | UI component library |
| Icons | lucide-react | 0.368.x | SVG icon components |
| CSS | Vanilla CSS | — | Custom design system (light theme) |
| Fonts | Google Fonts | — | DM Sans, Inter, Outfit |
| Backend framework | FastAPI | ≥0.104 | Async Python REST API |
| ASGI server | Uvicorn | ≥0.24 | Production-grade ASGI server |
| ML framework | PyTorch | 2.5.1 (CPU) | Tensor computation engine |
| ML model library | Transformers (HF) | ≥4.36 | Model loading + inference |
| Vision-Language model | BLIP (base) | — | Image captioning + VQA |
| Text-to-Speech | edge-tts | 7.2.8 | Microsoft neural TTS |
| Translation | googletrans | 4.0.0-rc1 | Free Google Translate wrapper |
| Image processing | Pillow | ≥10.0 | PIL image manipulation |
| Containerization | Docker | Multi-stage | Production deployment |

---

## 6. Improvements Needed

### 🔴 Critical / High Priority

| # | Area | Issue | Suggested Fix |
|---|------|-------|---------------|
| 1 | **Performance** | CPU inference takes 3–8s per request; unacceptable for production | Migrate to GPU inference (CUDA) or use ONNX Runtime for 2–4x CPU speedup |
| 2 | **Performance** | Model is FP32 (~1GB RAM) | Apply INT8 dynamic quantization — cuts memory by 50% and speeds up CPU inference 2x |
| 3 | **Reliability** | `googletrans` is an unofficial library that frequently breaks | Replace with `deep-translator` or official Google Cloud Translation API |
| 4 | **Scalability** | Single-threaded model inference blocks all requests | Add a task queue (Celery/Redis or asyncio semaphore) for concurrent request handling |
| 5 | **Error handling** | 500 errors don't log tracebacks to the terminal | Add `traceback.print_exc()` or structured logging (e.g., `loguru`) in exception handlers |

### 🟡 Medium Priority

| # | Area | Issue | Suggested Fix |
|---|------|-------|---------------|
| 6 | **Model quality** | BLIP-base produces short, sometimes generic captions | Upgrade to `blip-image-captioning-large` (446M params) or BLIP-2 for richer descriptions |
| 7 | **VQA quality** | BLIP-base VQA answers are often single words | Use a dedicated VQA model like `blip-vqa-base` or BLIP-2 with a proper VQA head |
| 8 | **TTS** | Only 2 voices (1 English, 1 Hindi) | Add a voice selector in the UI with multiple voice options per language |
| 9 | **Languages** | Only English and Hindi supported | Add more languages — edge-tts supports 40+ languages natively |
| 10 | **Frontend** | No loading skeleton or progress indicator for long requests | Add a skeleton UI or progress bar with estimated time |
| 11 | **Frontend** | No offline/PWA support | Add a service worker + manifest.json for installable PWA |
| 12 | **Security** | No rate limiting on API endpoints | Add `slowapi` rate limiter to prevent abuse |

### 🟢 Low Priority / Nice to Have

| # | Area | Issue | Suggested Fix |
|---|------|-------|---------------|
| 13 | **DevOps** | No CI/CD pipeline | Add GitHub Actions for lint, test, build, and Docker image push |
| 14 | **Testing** | No unit or integration tests exist | Add pytest tests for caption_model, tts, translate modules |
| 15 | **Monitoring** | No application metrics or health monitoring | Add Prometheus metrics endpoint + structured JSON logging |
| 16 | **Caching** | Same image re-uploaded generates caption from scratch every time | Add an in-memory LRU cache keyed on image hash |
| 17 | **Accessibility** | Screen reader testing not documented | Test with NVDA/JAWS and document ARIA compliance |
| 18 | **Frontend** | Chat history lost on page reload | Persist chat history to localStorage |
| 19 | **Model** | Model downloads on first startup (990MB) | Pre-bake model weights into the Docker image |

---

## 7. Potential Upgrade Paths

### Option A: ONNX Runtime (Quick Win)
```
BLIP PyTorch → Export to ONNX → Run with onnxruntime
• 2-4x faster CPU inference
• No code architecture changes needed
• Same model quality
```

### Option B: BLIP-2 with Quantization (Best Quality)
```
BLIP-base → BLIP-2 (Salesforce/blip2-opt-2.7b)
• Dramatically better captions and VQA answers
• Requires 4-bit quantization (bitsandbytes) to fit in memory
• Needs GPU for reasonable inference speed
```

### Option C: Florence-2 (Microsoft, Modern Alternative)
```
BLIP-base → microsoft/Florence-2-base
• Multi-task: captioning, VQA, OCR, grounding all in one model
• Smaller and faster than BLIP-2
• Better zero-shot performance
```

### Option D: Cloud API (Fastest, Costliest)
```
Local BLIP → Google Cloud Vision API / Azure Computer Vision
• Sub-second latency
• No GPU/model hosting needed
• Pay-per-request pricing
```

---

## 8. File Map

```
Aabha/
├── backend/
│   ├── main.py              # FastAPI app, routes, CORS, static serving
│   ├── caption_model.py     # BLIP model loading, captioning, VQA
│   ├── tts.py               # Edge-TTS async streaming
│   ├── translate.py         # googletrans translation wrapper
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/app/
│   │   ├── layout.js        # Root layout, Google Fonts, metadata
│   │   ├── page.js          # Main UI: upload, results, chat, SVGs
│   │   └── globals.css      # Full design system (light theme)
│   ├── next.config.mjs      # API proxy rewrites, static export
│   └── package.json         # Node dependencies
├── Dockerfile               # Multi-stage build (Node + Python)
├── .gitignore
└── README.md
```

---

## 9. Key Design Decisions

1. **CPU-only PyTorch** — The project targets accessibility/demo use, not production scale. GPU would improve speed 10-50x but adds deployment complexity.

2. **edge-tts over gTTS** — Microsoft's neural TTS voices sound significantly more natural than Google's basic TTS. It's also free and doesn't require API keys.

3. **Static Next.js export** — The frontend is built as a static site (`output: 'export'`) so FastAPI can serve it directly. No need for a separate Node.js server in production.

4. **Inline SVGs** — All accessibility-themed illustrations are inline SVG components (not external files), ensuring zero additional network requests and full CSS animation control.

5. **googletrans (unofficial)** — Chosen for zero-config setup. The `translate.py` module is deliberately a thin wrapper so swapping to Google Cloud Translation API is a one-file change.

6. **Model loaded at startup** — BLIP is loaded once into memory via FastAPI's lifespan context manager. This avoids per-request model loading latency (~10s) at the cost of ~1.4GB persistent RAM.

---

*Last updated: July 14, 2026*
