"""
test_pipeline.py — Unit tests for the AI Caption Generator backend.

Tests cover:
  - BLIP caption generation
  - gTTS audio generation
  - Translation (googletrans)
  - API endpoint response structure

Run with: python -m pytest tests/ -v
"""

import io
import pytest
from PIL import Image

from caption_model import generate_caption, load_model
from tts import caption_to_audio
from translate import translate_caption


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture(scope="session", autouse=True)
def setup_model():
    """Load the BLIP model once for all tests."""
    load_model()


@pytest.fixture
def sample_image():
    """Create a simple test image (solid blue, 100x100)."""
    img = Image.new("RGB", (100, 100), color=(70, 130, 180))
    return img


@pytest.fixture
def sample_rgba_image():
    """Create an RGBA test image to verify mode conversion."""
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    return img


# ---------------------------------------------------------------------------
# Caption Model Tests
# ---------------------------------------------------------------------------
class TestCaptionModel:
    """Tests for BLIP image captioning."""

    def test_generates_nonempty_caption(self, sample_image):
        """A valid image should produce a non-empty caption string."""
        caption = generate_caption(sample_image)
        assert isinstance(caption, str)
        assert len(caption) > 0

    def test_handles_rgba_image(self, sample_rgba_image):
        """RGBA images should be auto-converted to RGB without errors."""
        caption = generate_caption(sample_rgba_image)
        assert isinstance(caption, str)
        assert len(caption) > 0

    def test_handles_small_image(self):
        """Very small images (1x1) should still produce a caption."""
        tiny = Image.new("RGB", (1, 1), color=(0, 0, 0))
        caption = generate_caption(tiny)
        assert isinstance(caption, str)


# ---------------------------------------------------------------------------
# TTS Tests
# ---------------------------------------------------------------------------
class TestTTS:
    """Tests for text-to-speech audio generation."""

    @pytest.mark.anyio
    async def test_returns_mp3_bytes(self):
        """A valid English caption should produce non-empty MP3 bytes."""
        audio = await caption_to_audio("A dog running in a park.", lang="en")
        assert isinstance(audio, bytes)
        assert len(audio) > 0

    @pytest.mark.anyio
    async def test_hindi_audio(self):
        """Hindi text should produce non-empty MP3 bytes with lang='hi'."""
        audio = await caption_to_audio("एक कुत्ता पार्क में दौड़ रहा है।", lang="hi")
        assert isinstance(audio, bytes)
        assert len(audio) > 0

    @pytest.mark.anyio
    async def test_marathi_audio(self):
        """Marathi text should produce non-empty MP3 bytes with lang='mr'."""
        audio = await caption_to_audio("एक कुत्रा बागेत धावत आहे.", lang="mr")
        assert isinstance(audio, bytes)
        assert len(audio) > 0

    @pytest.mark.anyio
    async def test_empty_text_raises_error(self):
        """Empty text should raise a ValueError."""
        with pytest.raises(ValueError, match="empty"):
            await caption_to_audio("")

    @pytest.mark.anyio
    async def test_whitespace_only_raises_error(self):
        """Whitespace-only text should raise a ValueError."""
        with pytest.raises(ValueError, match="empty"):
            await caption_to_audio("   ")


# ---------------------------------------------------------------------------
# Translation Tests
# ---------------------------------------------------------------------------
class TestTranslation:
    """Tests for English-to-Hindi/Marathi translation."""

    def test_translates_to_hindi(self):
        """An English sentence should produce non-empty Hindi text."""
        result = translate_caption("A cat sitting on a mat.", target_lang="hi")
        assert isinstance(result, str)
        assert len(result) > 0
        # The result should be different from the input (it's translated)
        assert result != "A cat sitting on a mat."

    def test_translates_to_marathi(self):
        """An English sentence should produce non-empty Marathi text."""
        result = translate_caption("A cat sitting on a mat.", target_lang="mr")
        assert isinstance(result, str)
        assert len(result) > 0
        # The result should be different from the input (it's translated)
        assert result != "A cat sitting on a mat."

    def test_empty_text_raises_error(self):
        """Empty text should raise a ValueError."""
        with pytest.raises(ValueError, match="empty"):
            translate_caption("")


# ---------------------------------------------------------------------------
# Integration Test
# ---------------------------------------------------------------------------
class TestEndToEnd:
    """Integration test: image → caption → audio."""

    @pytest.mark.anyio
    async def test_full_pipeline_english(self, sample_image):
        """Full pipeline in English should produce a caption and audio."""
        caption = generate_caption(sample_image)
        assert len(caption) > 0

        audio = await caption_to_audio(caption, lang="en")
        assert len(audio) > 0

    @pytest.mark.anyio
    async def test_full_pipeline_hindi(self, sample_image):
        """Full pipeline in Hindi should produce a translated caption and audio."""
        caption = generate_caption(sample_image)
        translated = translate_caption(caption, target_lang="hi")
        assert len(translated) > 0

        audio = await caption_to_audio(translated, lang="hi")
        assert len(audio) > 0

    @pytest.mark.anyio
    async def test_full_pipeline_marathi(self, sample_image):
        """Full pipeline in Marathi should produce a translated caption and audio."""
        caption = generate_caption(sample_image)
        translated = translate_caption(caption, target_lang="mr")
        assert len(translated) > 0

        audio = await caption_to_audio(translated, lang="mr")
        assert len(audio) > 0
