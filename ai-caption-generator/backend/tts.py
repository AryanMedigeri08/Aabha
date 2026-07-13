"""
tts.py — Text-to-Speech Module

Wraps gTTS (Google Text-to-Speech) to convert caption text into MP3 audio bytes.
All audio is generated fully in-memory — no files are written to disk (privacy requirement).
"""

import io
from gtts import gTTS


def caption_to_audio(text: str, lang: str = "en") -> bytes:
    """
    Convert a caption string to MP3 audio bytes using gTTS.

    Args:
        text: The caption text to speak. Must be non-empty.
        lang: Language code — 'en' for English, 'hi' for Hindi.

    Returns:
        Raw MP3 audio bytes (can be sent directly to the browser or saved).

    Raises:
        ValueError: If text is empty or whitespace-only.
        RuntimeError: If gTTS fails (e.g., network error, unsupported language).
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate audio from empty text.")

    try:
        # Create gTTS object with the correct language
        tts = gTTS(text=text, lang=lang)

        # Write audio to an in-memory buffer (no disk I/O)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return audio_buffer.read()

    except Exception as e:
        raise RuntimeError(f"Text-to-speech generation failed: {str(e)}")
