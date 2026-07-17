"""
main.py — FastAPI Backend for AI Caption Generator

Serves the REST API that the Next.js frontend calls:
  POST /api/caption   — Upload image, get caption + audio back
  GET  /api/health    — Health check

The BLIP model is loaded once at startup and kept in memory.
"""

import base64
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

from multi_caption_model import load_models, generate_all_captions, compile_captions, generate_answer
from tts import caption_to_audio
from translate import translate_caption


# ---------------------------------------------------------------------------
# App lifespan: load the models once when the server starts
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the captioning models at startup, clean up on shutdown."""
    load_models()
    yield
    # Cleanup (if needed) goes here


# ---------------------------------------------------------------------------
# Create the FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Caption Generator API",
    description="Accessible image-to-speech captioning API powered by BLIP",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Next.js frontend (localhost:3000 for dev, any origin for deploy)
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    """Health check endpoint — returns 200 if the server and models are ready."""
    return {
        "status": "healthy",
        "models": ["blip-image-captioning-base", "git-base", "vit-gpt2-image-captioning"]
    }


@app.post("/api/caption")
async def create_caption(
    file: UploadFile = File(..., description="Image file (JPG, PNG, or WEBP, max 10MB)"),
    lang: str = Form(default="en", description="Target language: 'en', 'hi', or 'mr'"),
):
    """
    Upload an image and receive a caption + audio.

    - Validates file type and size.
    - Generates an English caption using BLIP.
    - Optionally translates to Hindi or Marathi.
    - Generates TTS audio for the final caption.
    - Returns everything as JSON (audio is base64-encoded).
    """

    # --- Validate file extension ---
    if file.filename:
        extension = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '.{extension}'. Please upload a JPG, PNG, or WEBP image.",
            )

    # --- Read and validate file size ---
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large ({len(file_bytes) / (1024*1024):.1f}MB). Maximum size is {MAX_FILE_SIZE_MB}MB.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # --- Open the image ---
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not open the file as an image. It may be corrupted or not a valid image format.",
        )

    # --- Generate English captions from multiple models ---
    try:
        captions = generate_all_captions(image)
        caption_en = compile_captions(captions)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Caption generation failed unexpectedly: {str(e)}",
        )

    # --- Translate if Hindi or Marathi is requested ---
    caption_translated = None
    if lang in ["hi", "mr"]:
        try:
            caption_translated = translate_caption(caption_en, target_lang=lang)
        except RuntimeError as e:
            # Translation failed — still return the English caption + a warning
            caption_translated = None
            print(f"Translation warning: {e}")

    # --- Determine the final caption for TTS ---
    final_caption = caption_translated if caption_translated else caption_en
    tts_lang = lang if (lang in ["hi", "mr"] and caption_translated) else "en"

    # --- Generate audio ---
    try:
        audio_bytes = await caption_to_audio(final_caption, lang=tts_lang)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    except (ValueError, RuntimeError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio generation failed: {str(e)}",
        )

    # --- Return the response ---
    return JSONResponse(
        content={
            "caption_en": caption_en,  # Fused consensus description
            "captions": captions,      # Individual model descriptions
            "caption_translated": caption_translated,
            "lang": tts_lang,
            "audio_base64": audio_base64,
            "audio_format": "mp3",
        }
    )


@app.post("/api/chat")
async def chat_image(
    file: UploadFile = File(..., description="Image file (JPG, PNG, or WEBP, max 10MB)"),
    question: str = Form(..., description="User question about the image"),
    lang: str = Form(default="en", description="Target language: 'en', 'hi', or 'mr'"),
):
    """
    Ask a question about an image and receive a translated answer + spoken audio.
    """
    # --- Validate file extension ---
    if file.filename:
        extension = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '.{extension}'. Please upload a JPG, PNG, or WEBP image.",
            )

    # --- Read and validate file size ---
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large. Maximum size is {MAX_FILE_SIZE_MB}MB.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # --- Open the image ---
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not open the file as an image.",
        )

    # --- Generate answer ---
    try:
        answer_en = generate_answer(image, question)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Visual question answering failed: {str(e)}",
        )

    # --- Translate to Hindi or Marathi if requested ---
    answer_translated = None
    if lang in ["hi", "mr"]:
        try:
            answer_translated = translate_caption(answer_en, target_lang=lang)
        except RuntimeError as e:
            answer_translated = None
            print(f"Translation warning: {e}")

    # --- Determine the final answer for TTS ---
    final_answer = answer_translated if answer_translated else answer_en
    tts_lang = lang if (lang in ["hi", "mr"] and answer_translated) else "en"

    # --- Generate audio ---
    try:
        audio_bytes = await caption_to_audio(final_answer, lang=tts_lang)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    except (ValueError, RuntimeError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio generation failed: {str(e)}",
        )

    # --- Return response ---
    return JSONResponse(
        content={
            "answer_en": answer_en,
            "answer_translated": answer_translated,
            "lang": tts_lang,
            "audio_base64": audio_base64,
            "audio_format": "mp3",
        }
    )


# ---------------------------------------------------------------------------
# Serve Static Frontend Files (Production Build)
# ---------------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles

# Resolve the absolute path to the frontend build folder (out/)
frontend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "out"
)

# Mount the Next.js static build if it exists
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


# ---------------------------------------------------------------------------
# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
