"""
translate.py — Translation Abstraction Layer

Provides a translate_caption() function that translates English text to a
target language (currently Hindi). Uses googletrans as the free backend.

DESIGN: This module is intentionally kept as a thin abstraction so that
swapping in the official Google Cloud Translate API later is a one-line change
(replace the _translate_with_googletrans function body).
"""

from googletrans import Translator

# Reuse a single Translator instance across calls
_translator = Translator()


def translate_caption(text: str, target_lang: str = "hi") -> str:
    """
    Translate a caption from English to the target language.

    Args:
        text: English caption text to translate.
        target_lang: Target language code (default: 'hi' for Hindi).

    Returns:
        Translated text string.

    Raises:
        ValueError: If text is empty.
        RuntimeError: If translation fails.
    """
    if not text or not text.strip():
        raise ValueError("Cannot translate empty text.")

    try:
        return _translate_with_googletrans(text, target_lang)
    except Exception as e:
        raise RuntimeError(f"Translation failed: {str(e)}")


def _translate_with_googletrans(text: str, target_lang: str) -> str:
    """
    Internal: translate using the free googletrans library.

    To swap in the official Google Cloud Translate API, replace this
    function body with the google-cloud-translate client call.
    For example:
        from google.cloud import translate_v2 as translate
        client = translate.Client()
        result = client.translate(text, target_language=target_lang)
        return result['translatedText']
    """
    result = _translator.translate(text, src="en", dest=target_lang)
    return result.text
