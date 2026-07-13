# Product Requirements Document (PRD)
## AI Caption Generator — Accessible Image-to-Speech App

**Project type:** 8-week B.Tech internship / academic project
**Owner:** [Student Name]
**Status:** Draft v1.0

---

## 1. Problem Statement

Visually impaired users cannot independently understand the content of images shared on the web, in documents, or on social media. Existing screen readers can read alt-text, but most images online have no alt-text or poor alt-text. There is a need for a lightweight tool that takes any image and produces a spoken, natural-language description of it, with no manual tagging required.

## 2. Goal

Build a web application that:
1. Accepts an uploaded image from the user.
2. Generates a natural-language caption describing the image content using a vision-language transformer model.
3. Converts that caption into speech (audio) so it can be listened to.
4. Optionally translates the caption into Hindi for regional accessibility.
5. Is deployable as a free, public web app.

## 3. Target Users

- Visually impaired users who want a quick description of an image (photo, document, scene) read aloud to them.
- Caregivers/family members generating descriptions on behalf of a visually impaired user.
- Secondary audience: developers/students who want to see a working example of a multimodal transformer pipeline.

## 4. Non-Goals (Out of Scope)

- Real-time video captioning (only static images in v1).
- Support for languages beyond English and Hindi.
- Mobile native app (web app only, though mobile-responsive).
- Training or fine-tuning a custom model (BLIP is used pretrained/off-the-shelf).
- Handling extremely large batch uploads (single-image workflow only).

## 5. User Stories

| # | As a... | I want to... | So that... |
|---|---------|--------------|------------|
| 1 | Visually impaired user | Upload a photo | I understand what is in the picture without asking someone |
| 2 | Visually impaired user | Hear the description read aloud | I don't need to read text on screen |
| 3 | Hindi-speaking user | Get the caption in Hindi | I understand it in my preferred language |
| 4 | Any user | Drag and drop an image | The upload process is fast and simple |
| 5 | Accessibility auditor | Navigate the app with a screen reader | The app itself is usable, not just its output |

## 6. Functional Requirements

### 6.1 Core Captioning
- FR-1: The app must accept image uploads in JPG, PNG, and WEBP formats.
- FR-2: The app must generate a caption using the BLIP image-captioning model (`Salesforce/blip-image-captioning-base` or `-large`).
- FR-3: Caption generation must complete in under ~10 seconds on CPU for a typical image.
- FR-4: The generated caption must be displayed as text on screen.

### 6.2 Audio Output
- FR-5: The app must convert the generated caption to speech using gTTS (Google Text-to-Speech).
- FR-6: The app must play the generated audio directly in the browser via an embedded audio player.
- FR-7: The user must be able to replay or download the audio file.

### 6.3 Multilingual Support
- FR-8: The app must offer an option to translate the English caption into Hindi using the Google Translate API.
- FR-9: When Hindi is selected, both the Hindi text and a Hindi audio clip (via gTTS `lang='hi'`) must be produced.

### 6.4 Upload & UI
- FR-10: The app must support drag-and-drop image upload as well as a traditional file picker.
- FR-11: The app must show a preview of the uploaded image alongside its caption.
- FR-12: The UI must be usable via keyboard navigation and screen reader (ARIA labels on all interactive elements).

### 6.5 Deployment
- FR-13: The final app must be deployed publicly on Hugging Face Spaces with a shareable URL.

## 7. Non-Functional Requirements

- **Performance:** Caption generation should not block the UI; a loading indicator must be shown.
- **Cost:** The entire pipeline must run on free tiers (BLIP on CPU, gTTS free, Hugging Face Spaces free tier).
- **Accessibility:** WCAG 2.1 AA-inspired checklist — alt text on images, sufficient color contrast, screen-reader-friendly labels, keyboard operability.
- **Reliability:** Graceful error handling for unsupported file types, oversized files (>10MB), or failed translation/audio calls.
- **Privacy:** Uploaded images must not be persisted beyond the session; no user data stored server-side.

## 8. Success Metrics

| Metric | Target |
|--------|--------|
| Caption generation success rate | >95% of valid image uploads produce a caption |
| End-to-end latency (upload → audio playback ready) | <15 seconds on CPU |
| Accessibility audit | Passes manual screen-reader walkthrough (NVDA/VoiceOver) with no blocking issues |
| Demo "wow" factor | Live demo successfully captions + narrates 5 diverse test images (face, scene, document, object, group photo) |

## 9. Timeline (8 Weeks)

| Weeks | Phase | Key Output |
|-------|-------|-----------|
| 1–2 | Foundation | BLIP running in Colab; understand image→text pipeline |
| 3–4 | Audio layer | gTTS integrated; caption → audio file → browser playback |
| 5–6 | UI + language | Drag-and-drop upload; Hindi caption via Google Translate API |
| 7–8 | Polish & deploy | Testing on diverse images, accessibility audit, deployment to HF Spaces |

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| BLIP inference too slow on CPU | Use `blip-image-captioning-base` (smaller) instead of `-large`; cache model in memory |
| Google Translate API rate limits / requires key | Use `googletrans` (unofficial, free) as fallback if official API quota is a blocker |
| Captions inaccurate for certain image types (faces, documents) | Explicitly test and document limitations in final report; set user expectations |
| Accessibility claims not actually validated | Run a real screen-reader walkthrough (NVDA or VoiceOver) before final submission |

## 11. Deliverables

1. Deployed web app on Hugging Face Spaces.
2. Source code repository (Streamlit app + supporting modules).
3. Final report covering methodology, testing results, and accessibility findings.
4. This PRD, the System Architecture doc, and the Technical Implementation doc.
