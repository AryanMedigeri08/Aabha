"""
multi_caption_model.py — Multi-Model Image Captioning and VQA Module

Loads Salesforce/blip-image-captioning-base, microsoft/git-base,
and nlpconnect/vit-gpt2-image-captioning models. Performs parallel inference
and implements caption compilation.
"""

import time
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoProcessor, AutoModelForCausalLM,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
)

# Device configuration (detect GPU/iGPU where CUDA is available, otherwise CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Module-level globals for lazy/cached loading
_models_loaded = False
_blip_processor = None
_blip_model = None

_git_processor = None
_git_model = None

_vit_gpt2_extractor = None
_vit_gpt2_tokenizer = None
_vit_gpt2_model = None


def load_models():
    """
    Load the processor and model parameters for BLIP, GIT, and ViT-GPT2.
    """
    global _models_loaded
    global _blip_processor, _blip_model
    global _git_processor, _git_model
    global _vit_gpt2_extractor, _vit_gpt2_tokenizer, _vit_gpt2_model

    if _models_loaded:
        print("Models already loaded in cache.")
        return

    print(f"Loading multiple models onto device: {DEVICE}...")
    start_time = time.time()

    # 1. Load BLIP
    print("Loading BLIP...")
    _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    _blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        use_safetensors=True
    ).to(DEVICE)
    _blip_model.eval()

    # 2. Load GIT
    print("Loading Microsoft GIT...")
    _git_processor = AutoProcessor.from_pretrained("microsoft/git-base")
    _git_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/git-base"
    ).to(DEVICE)
    _git_model.eval()

    # 3. Load ViT-GPT2
    print("Loading ViT-GPT2...")
    _vit_gpt2_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    _vit_gpt2_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    _vit_gpt2_model = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    ).to(DEVICE)
    _vit_gpt2_model.eval()

    _models_loaded = True
    print(f"All models loaded successfully in {time.time() - start_time:.2f} seconds.")


def _run_blip(image: Image.Image) -> str:
    try:
        inputs = _blip_processor(image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output_ids = _blip_model.generate(**inputs, max_new_tokens=40)
        caption = _blip_processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        print(f"BLIP inference error: {e}")
        return ""


def _run_git(image: Image.Image) -> str:
    try:
        inputs = _git_processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output_ids = _git_model.generate(
                pixel_values=inputs.pixel_values,
                max_new_tokens=40
            )
        caption = _git_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return caption.strip()
    except Exception as e:
        print(f"GIT inference error: {e}")
        return ""


def _run_vit_gpt2(image: Image.Image) -> str:
    try:
        # Preprocess image
        pixel_values = _vit_gpt2_extractor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
        # Generate caption
        with torch.no_grad():
            output_ids = _vit_gpt2_model.generate(
                pixel_values,
                max_new_tokens=40,
                num_beams=4
            )
        caption = _vit_gpt2_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return caption.strip()
    except Exception as e:
        print(f"ViT-GPT2 inference error: {e}")
        return ""


def generate_all_captions(image: Image.Image) -> dict:
    """
    Run parallel caption generation on all three models.
    """
    if not _models_loaded:
        raise RuntimeError("Models are not loaded. Call load_models() first.")

    # Convert to RGB mode if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize if extremely small to avoid processor size errors
    if image.width < 32 or image.height < 32:
        image = image.resize((224, 224), Image.Resampling.BILINEAR)

    print("Running parallel caption generation across BLIP, GIT, and ViT-GPT2...")
    start_time = time.time()

    # Run inference concurrently in separate threads
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_blip = executor.submit(_run_blip, image)
        future_git = executor.submit(_run_git, image)
        future_vit = executor.submit(_run_vit_gpt2, image)

        blip_caption = future_blip.result()
        git_caption = future_git.result()
        vit_caption = future_vit.result()

    print(f"Parallel inference completed in {time.time() - start_time:.2f} seconds.")
    return {
        "blip": blip_caption,
        "git": git_caption,
        "vit_gpt2": vit_caption
    }


def compile_captions(captions: dict) -> str:
    """
    Consensus Compiler (Option A): Heuristically compiles multiple captions
    into one cohesive description, removing duplicates and merging information.
    """
    valid_captions = [cap.strip() for cap in captions.values() if cap and cap.strip()]
    if not valid_captions:
        return "An image."

    # Normalization: lower-case and strip punctuation
    sentences = []
    for cap in valid_captions:
        # Standardize capitalization and punctuation
        s = cap.lower().rstrip(".").strip()
        if s:
            sentences.append(s)

    # Simple consensus compilation:
    # 1. Identify distinct key concepts or sentences.
    # 2. Filter out sentences that are sub-strings or extremely similar to existing ones.
    unique_sentences = []
    for s in sentences:
        is_redundant = False
        for existing in unique_sentences:
            # If the sentence is a substring of an already selected one, or vice-versa, skip it
            if s in existing or existing in s:
                # Keep the longer, more informative sentence
                if len(s) > len(existing):
                    unique_sentences.remove(existing)
                    unique_sentences.append(s)
                is_redundant = True
                break
        if not is_redundant:
            unique_sentences.append(s)

    # Reformat sentences nicely: capitalize first letter and join with periods
    compiled_parts = [s[0].upper() + s[1:] for s in unique_sentences]
    return ". ".join(compiled_parts) + "."


def generate_answer(image: Image.Image, question: str) -> str:
    """
    Generate an answer to a question about the given PIL Image using the loaded BLIP model.
    """
    if not _models_loaded:
        raise RuntimeError("Models are not loaded. Call load_models() first.")

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    try:
        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Safeguard: if image size is extremely small, resize it to at least 224x224
        if image.width < 32 or image.height < 32:
            image = image.resize((224, 224), Image.Resampling.BILINEAR)

        # Format question prompt for BLIP
        prompt = f"Question: {question.strip()} Answer:"

        # Preprocess both image and text prompt
        inputs = _blip_processor(image, text=prompt, return_tensors="pt").to(DEVICE)

        # Generate answer tokens
        with torch.no_grad():
            output_ids = _blip_model.generate(**inputs, max_new_tokens=40)

        # Decode token IDs
        answer = _blip_processor.decode(output_ids[0], skip_special_tokens=True)

        # Clean prompt details out if returned by model
        lower_answer = answer.lower()
        if "answer:" in lower_answer:
            answer = answer.split("answer:")[-1]
        elif "answer" in lower_answer:
            answer = answer.split("answer")[-1]

        return answer.strip()

    except Exception as e:
        raise ValueError(f"Failed to generate answer: {str(e)}")



# Standalone test block
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING MULTI-MODEL INFERENCE LAYER")
    print("=" * 60)

    # 1. Test model loading
    load_models()

    # 2. Test inference on a dummy image
    dummy_image = Image.new("RGB", (224, 224), color="blue")
    results = generate_all_captions(dummy_image)
    print("\nInference Results:")
    for model_name, caption in results.items():
        print(f"  {model_name}: '{caption}'")

    # 3. Test compile heuristics
    test_caps = {
        "blip": "a blue square on a white background",
        "git": "a blue square",
        "vit_gpt2": "a solid blue square background image"
    }
    compiled = compile_captions(test_caps)
    print(f"\nTest Captions:")
    for m, c in test_caps.items():
        print(f"  {m}: {c}")
    print(f"Compiled: '{compiled}'")
