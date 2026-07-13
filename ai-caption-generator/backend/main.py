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

from caption_model import generate_caption, load_model
from tts import caption_to_audio
from translate import translate_caption


# ---------------------------------------------------------------------------
# App lifespan: load the BLIP model once when the server starts
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the BLIP model at startup, clean up on shutdown."""
    load_model()
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
    """Health check endpoint — returns 200 if the server and model are ready."""
    return {"status": "healthy", "model": "blip-image-captioning-base"}


@app.post("/api/caption")
async def create_caption(
    file: UploadFile = File(..., description="Image file (JPG, PNG, or WEBP, max 10MB)"),
    lang: str = Form(default="en", description="Target language: 'en' or 'hi'"),
):
    """
    Upload an image and receive a caption + audio.

    - Validates file type and size.
    - Generates an English caption using BLIP.
    - Optionally translates to Hindi.
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

    # --- Generate English caption ---
    try:
        caption_en = generate_caption(image)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Caption generation failed unexpectedly: {str(e)}",
        )

    # --- Translate if Hindi is requested ---
    caption_translated = None
    if lang == "hi":
        try:
            caption_translated = translate_caption(caption_en, target_lang="hi")
        except RuntimeError as e:
            # Translation failed — still return the English caption + a warning
            caption_translated = None
            print(f"Translation warning: {e}")

    # --- Determine the final caption for TTS ---
    final_caption = caption_translated if caption_translated else caption_en
    tts_lang = "hi" if (lang == "hi" and caption_translated) else "en"

    # --- Generate audio ---
    try:
        audio_bytes = caption_to_audio(final_caption, lang=tts_lang)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    except (ValueError, RuntimeError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio generation failed: {str(e)}",
        )

    # --- Return the response ---
    return JSONResponse(
        content={
            "caption_en": caption_en,
            "caption_translated": caption_translated,
            "lang": tts_lang,
            "audio_base64": audio_base64,
            "audio_format": "mp3",
        }
    )


# ---------------------------------------------------------------------------
# Serve Static Frontend Files
# ---------------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles

# Resolve the absolute path to the frontend folder
frontend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"
)

# Ensure the frontend folder exists before mounting
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


# ---------------------------------------------------------------------------
# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
