"""
tts.py — Text-to-Speech Module using Edge Neural TTS

Converts caption text into high-quality MP3 audio bytes using Microsoft Edge's neural voices.
"""

import edge_tts


async def caption_to_audio(text: str, lang: str = "en") -> bytes:
    """
    Convert a caption string to MP3 audio bytes using edge-tts.

    Args:
        text: The caption text to speak. Must be non-empty.
        lang: Language code — 'en' for English, 'hi' for Hindi.

    Returns:
        Raw MP3 audio bytes.

    Raises:
        ValueError: If text is empty or whitespace-only.
        RuntimeError: If audio generation fails.
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate audio from empty text.")

    try:
        # Choose natural neural voices
        # English: en-US-Multilingual-RyanNeural or en-US-AriaNeural
        # Hindi: hi-IN-MadhurNeural or hi-IN-SwararaNeural
        voice = "en-US-Multilingual-RyanNeural" if lang == "en" else "hi-IN-MadhurNeural"
        
        communicate = edge_tts.Communicate(text, voice)
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]
        
        if not audio_bytes:
            raise RuntimeError("Generated audio stream was empty.")
            
        return audio_bytes

    except Exception as e:
        raise RuntimeError(f"Text-to-speech generation failed: {str(e)}")
