# Implementation Prompt for Claude Opus 4.6
## Build: AI Caption Generator — Accessible Image-to-Speech Web App

Copy everything below into Claude (Opus 4.6 / Claude Code) as a single message to build the full project in one pass.

---

```
You are building a complete, working web application called "AI Caption Generator" —
an accessibility tool that lets a user upload any image, generates a natural-language
caption using a vision-language transformer, and reads that caption aloud as audio.
It is designed primarily for visually impaired users. This is an 8-week B.Tech
internship project; build it as a polished, working v1, not a toy prototype.

## PROJECT CONTEXT

- Target users: visually impaired users (primary), caregivers, and reviewers/evaluators
  of the project.
- Core value prop: upload image → hear a spoken description of what's in it →
  optionally hear it in Hindi.
- Must run entirely on free tooling: BLIP on CPU, gTTS (free), Streamlit,
  deployable on Hugging Face Spaces.
- No model training or fine-tuning — use BLIP pretrained, off the shelf.
- No persistent storage of user images or audio — everything stays in memory for
  the session only (privacy requirement).

## TECH STACK (use exactly this stack)

- Frontend + app framework: Streamlit
- Captioning model: Salesforce/blip-image-captioning-base (HuggingFace Transformers, PyTorch, CPU)
- Image handling: PIL / Pillow
- Text-to-speech: gTTS
- Translation (English → Hindi): googletrans (googletrans==4.0.0-rc1) as the free
  fallback if a paid Google Translate API key is not available — implement it so
  that swapping in the official Google Cloud Translate API later is a one-line change
- Language: Python 3.10+
- Deployment target: Hugging Face Spaces (Streamlit SDK)

## FUNCTIONAL REQUIREMENTS — BUILD ALL OF THESE

1. Image upload
   - Support drag-and-drop AND a traditional file picker (Streamlit's file_uploader
     supports both natively — use it).
   - Accept JPG, JPEG, PNG, WEBP.
   - Reject files over 10MB with a clear, friendly error message.
   - Show a preview of the uploaded image with proper alt text in the DOM.

2. Captioning
   - Load Salesforce/blip-image-captioning-base once and cache it with
     @st.cache_resource so it isn't reloaded per interaction.
   - Generate an English caption for the uploaded image.
   - Show a loading spinner while inference runs.
   - Handle and surface errors gracefully (e.g., corrupted image file) instead of crashing.

3. Language toggle
   - Radio button or toggle: English / Hindi.
   - When Hindi is selected, translate the English caption to Hindi using googletrans,
     wrapped in a small abstraction (e.g., a `translate_caption(text, target_lang)`
     function) so the translation backend can be swapped later without touching
     the rest of the app.

4. Text-to-speech
   - Convert the final caption (English or Hindi) to speech using gTTS, with the
     correct lang code ('en' or 'hi').
   - Generate the audio fully in-memory (io.BytesIO) — do NOT write it to disk.
   - Render it with st.audio() so it plays directly in the browser.
   - Provide a download button for the audio file (caption.mp3).

5. Accessibility of the app itself (not just its output)
   - All interactive elements must be reachable via keyboard (Tab order).
   - Add descriptive labels/help text to inputs so a screen reader announces them
     sensibly (upload button, language toggle, audio player).
   - Use a high-contrast, readable theme.
   - No flashing/auto-moving UI elements.

6. Privacy
   - No database, no file persistence to disk, no logging of uploaded image contents.
   - State should reset cleanly between sessions/users.

## NON-FUNCTIONAL REQUIREMENTS

- End-to-end latency (upload → audio ready to play) should be under ~15 seconds on
  CPU for a typical image. Use the "base" BLIP checkpoint, not "large", to keep this
  fast.
- The whole thing must run for free: no GPU dependency, no paid API keys required
  by default.
- Code should be clean, modular, and commented well enough that a beginner
  (2nd-year B.Tech student) could read and explain every part of it.

## PROJECT STRUCTURE — CREATE THIS LAYOUT

```
ai-caption-generator/
├── app.py                  # Main Streamlit app (entry point)
├── caption_model.py         # BLIP loading + generate_caption()
├── tts.py                    # gTTS wrapper: caption_to_audio()
├── translate.py              # translate_caption() abstraction
├── requirements.txt
├── README.md                 # Includes HF Spaces front-matter (sdk: streamlit, app_file: app.py)
└── tests/
    └── test_pipeline.py      # Basic tests: caption generation runs, TTS returns bytes, etc.
```

## DELIVERABLES FOR THIS SESSION

1. All files above, fully implemented and runnable with `streamlit run app.py`.
2. requirements.txt pinned with CPU-only torch install instructions in a comment.
3. README.md with:
   - Project description
   - Setup instructions (venv, pip install, run command)
   - Hugging Face Spaces deployment instructions (YAML front-matter + git push steps)
   - Known limitations section (e.g., BLIP is not OCR, may caption text-heavy
     images poorly; not designed for real-time video)
4. A short section at the end of your response summarizing:
   - What you built
   - Any assumptions you made
   - What I should test manually before considering this done (include the test
     matrix: face/portrait photo, outdoor scene, document/text-heavy image, group
     photo, low-light/blurry image)

## IMPORTANT CONSTRAINTS

- Do not add authentication, user accounts, or a database — out of scope for v1.
- Do not implement video or multi-image batch captioning — out of scope for v1.
- Do not silently swallow errors — surface friendly messages in the UI.
- Prefer simplicity and readability over cleverness; this is a learning project as
  much as a deliverable.

Build the complete, working implementation now.
```

---

### Notes on using this prompt

- This prompt assumes Claude Opus 4.6 has code-execution / file-creation capability (e.g. Claude Code, Cowork, or the Claude.ai coding environment). If you're pasting into a plain chat interface without file tools, ask it to output each file in a separate fenced code block instead.
- Pair this with the `PRD.md`, `System_architecture.md`, and `Technical_Implementation.md` files already generated — you can attach those alongside this prompt for even more grounding, though the prompt above is self-contained.
- If you later want to swap `googletrans` for the official paid Google Cloud Translate API, only `translate.py` needs to change.
