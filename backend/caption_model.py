"""
caption_model.py — BLIP Image Captioning Module

Loads the Salesforce/blip-image-captioning-base model once and provides
a generate_caption() function that takes a PIL Image and returns an
English caption string.

Uses the "base" checkpoint (not "large") to keep CPU inference under ~10 seconds.
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Module-level globals — populated by load_model()
_processor = None
_model = None


def load_model():
    """
    Load the BLIP processor and model into memory.
    Call this once at app startup. The model stays in memory for all
    subsequent requests (equivalent to Streamlit's @st.cache_resource).
    """
    global _processor, _model

    print("Loading BLIP model (this may take a minute on first run)...")
    _processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    _model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    # Set model to eval mode — disables dropout, etc.
    _model.eval()
    print("BLIP model loaded successfully.")


def generate_caption(image: Image.Image) -> str:
    """
    Generate an English caption for the given PIL Image.

    Args:
        image: A PIL Image in RGB mode.

    Returns:
        A natural-language English caption string describing the image.

    Raises:
        RuntimeError: If the model hasn't been loaded yet (call load_model() first).
        ValueError: If the image is invalid or cannot be processed.
    """
    if _processor is None or _model is None:
        raise RuntimeError(
            "BLIP model not loaded. Call load_model() at app startup."
        )

    try:
        # Convert to RGB if not already (handles RGBA, grayscale, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess: resize + normalize → PyTorch tensor
        inputs = _processor(image, return_tensors="pt")

        # Generate caption tokens (max_new_tokens=40 keeps it concise and fast)
        output_ids = _model.generate(**inputs, max_new_tokens=40)

        # Decode token IDs → human-readable caption string
        caption = _processor.decode(output_ids[0], skip_special_tokens=True)

        return caption.strip()

    except Exception as e:
        raise ValueError(f"Failed to generate caption: {str(e)}")
