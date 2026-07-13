# Technical Implementation Plan
## AI Caption Generator — Accessible Image-to-Speech App

---

## 1. Environment Setup

```bash
# Create environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Core dependencies
pip install streamlit torch torchvision transformers pillow gtts googletrans==4.0.0-rc1
```

**requirements.txt (for Hugging Face Spaces deployment):**
```
streamlit
torch --index-url https://download.pytorch.org/whl/cpu
transformers
Pillow
gTTS
googletrans==4.0.0-rc1
```

---

## 2. Weeks 1–2: Foundation — BLIP Captioning Pipeline

**Goal:** Run BLIP on sample images in Colab and understand the image-to-text pipeline output.

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image: Image.Image) -> str:
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=40)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

img = Image.open("sample.jpg").convert("RGB")
print(generate_caption(img))
```

**Tasks:**
- Load and test BLIP on 10+ diverse sample images (people, objects, scenes).
- Log inference time per image on CPU (Colab and local).
- Read the BLIP paper/model card to understand encoder-decoder + cross-attention design.
- Note failure cases (blurry images, multiple subjects, text-heavy images).

---

## 3. Weeks 3–4: Audio Layer — Text-to-Speech Integration

**Goal:** Add gTTS, generate an audio file from the caption, and play it in the browser.

```python
from gtts import gTTS
import io

def caption_to_audio(text: str, lang: str = "en") -> bytes:
    tts = gTTS(text=text, lang=lang)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()

audio_bytes = caption_to_audio("A dog running across a grassy field.")
with open("caption_audio.mp3", "wb") as f:
    f.write(audio_bytes)
```

**Streamlit integration:**
```python
import streamlit as st

st.audio(audio_bytes, format="audio/mp3")
st.download_button("Download audio", data=audio_bytes, file_name="caption.mp3")
```

**Tasks:**
- Wire caption output directly into `caption_to_audio()`.
- Confirm playback works in-browser (not just saved to disk).
- Handle empty/very long captions gracefully (truncate or split if needed).

---

## 4. Weeks 5–6: UI + Multilingual Support

**Goal:** Drag-and-drop image upload; add a Hindi caption option via the Google Translate API.

```python
from googletrans import Translator

translator = Translator()

def translate_to_hindi(text: str) -> str:
    result = translator.translate(text, src="en", dest="hi")
    return result.text
```

**Streamlit UI:**
```python
import streamlit as st
from PIL import Image

st.title("AI Caption Generator — for Accessibility")

uploaded_file = st.file_uploader(
    "Drag and drop an image, or click to browse",
    type=["jpg", "jpeg", "png", "webp"]
)

language = st.radio("Caption language", ["English", "Hindi"], horizontal=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    with st.spinner("Generating caption..."):
        caption_en = generate_caption(image)

    if language == "Hindi":
        caption_final = translate_to_hindi(caption_en)
        lang_code = "hi"
    else:
        caption_final = caption_en
        lang_code = "en"

    st.subheader("Caption")
    st.write(caption_final)

    with st.spinner("Generating audio..."):
        audio_bytes = caption_to_audio(caption_final, lang=lang_code)

    st.audio(audio_bytes, format="audio/mp3")
    st.download_button("Download audio", data=audio_bytes, file_name="caption.mp3")
```

**Tasks:**
- Style drag-and-drop area for clarity (Streamlit's file_uploader natively supports drag-and-drop).
- Add loading spinners for both captioning and audio generation steps.
- Add ARIA-friendly labels / alt text for all UI elements (screen-reader compatibility).
- Test Hindi translation on 10+ varied captions for accuracy.

---

## 5. Weeks 7–8: Testing, Accessibility Audit & Deployment

### 5.1 Test Matrix

| Image Type | Expected Behavior | Notes |
|-----------|-------------------|-------|
| Human face/portrait | Caption describes person, general appearance | Avoid over-claiming identity |
| Outdoor scene | Caption describes setting, objects | |
| Document/text-heavy image | BLIP may caption poorly (not OCR) | Document as a known limitation |
| Group photo | Caption should mention multiple people | |
| Low-light/blurry image | Caption may be inaccurate — test explicitly | Document limitation |

### 5.2 Accessibility Audit Checklist

- [ ] All interactive elements reachable via Tab key
- [ ] Screen reader (NVDA or VoiceOver) announces upload button, language toggle, and audio player correctly
- [ ] Color contrast meets WCAG AA (verify with a contrast checker)
- [ ] Uploaded image has descriptive alt text in the DOM
- [ ] No flashing/moving elements that could disorient users

### 5.3 Deployment to Hugging Face Spaces

```bash
# In your project folder
git init
git remote add origin https://huggingface.co/spaces/<username>/ai-caption-generator
git add app.py requirements.txt
git commit -m "Initial deploy"
git push origin main
```

- Set the Space SDK to `streamlit` in `README.md` front-matter:
```yaml
---
title: AI Caption Generator
sdk: streamlit
app_file: app.py
---
```
- Verify the deployed app loads the BLIP model correctly on the free CPU tier (may take longer on first load — consider `@st.cache_resource` to avoid reloading per session).

### 5.4 Final Report Contents
1. Methodology (BLIP architecture, TTS, translation pipeline)
2. Test results across the image test matrix
3. Accessibility audit findings and fixes made
4. Known limitations (document-heavy images, non-English objects, etc.)
5. Future work (more languages, batch captioning, mobile app)

---

## 6. Performance Notes

- Use `blip-image-captioning-base` (not `-large`) to keep CPU inference under ~5–10 seconds per image.
- Cache the model with `@st.cache_resource` so it loads once per app lifetime, not per request:
```python
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()
```
- Limit uploaded file size (e.g., reject files >10MB) to keep memory usage predictable on the free HF Spaces tier.
