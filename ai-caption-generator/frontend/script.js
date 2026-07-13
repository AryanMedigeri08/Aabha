document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const btnGenerate = document.getElementById('btn-generate');
    const langEn = document.getElementById('lang-en');
    const langHi = document.getElementById('lang-hi');
    const errorMessage = document.getElementById('error-message');
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultsSection = document.getElementById('results-section');
    const imagePreview = document.getElementById('image-preview');
    const textCaptionEn = document.getElementById('text-caption-en');
    const captionTranslatedBlock = document.getElementById('caption-translated-block');
    const textCaptionTranslated = document.getElementById('text-caption-translated');
    const audioPlayer = document.getElementById('audio-player');
    const btnDownload = document.getElementById('btn-download');

    let selectedFile = null;
    const MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10MB
    const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

    // --- Keyboard Accessibility for Drag-and-Drop Area ---
    dropZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileInput.click();
        }
    });

    // --- File Input Click Handler ---
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // --- Drag and Drop Events ---
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-over');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
            fileInput.files = files; // Sync file input element
        }
    });

    // --- File Selection & Validation ---
    function handleFileSelect(file) {
        hideError();
        selectedFile = null;
        btnGenerate.disabled = true;
        btnGenerate.removeAttribute('aria-describedby');

        // Validate File Type
        if (!ALLOWED_TYPES.includes(file.type)) {
            showError(`Unsupported file format (${file.type || 'unknown'}). Please upload a JPEG, PNG, or WEBP image.`);
            return;
        }

        // Validate File Size
        if (file.size > MAX_SIZE_BYTES) {
            const sizeInMB = (file.size / (1024 * 1024)).toFixed(1);
            showError(`The file is too large (${sizeInMB}MB). Maximum allowed size is 10MB.`);
            return;
        }

        selectedFile = file;
        btnGenerate.disabled = false;
        btnGenerate.setAttribute('aria-label', `Generate caption and audio for ${file.name}`);

        // Update Dropzone Text
        const instructions = document.getElementById('upload-instructions');
        instructions.innerHTML = `Selected file: <span class="highlight">${file.name}</span>`;
    }

    // --- Generate Caption & Audio Request ---
    btnGenerate.addEventListener('click', async () => {
        if (!selectedFile) return;

        hideError();
        hideResults();
        showLoading(true);

        const formData = new FormData();
        formData.append('file', selectedFile);
        
        const language = langHi.checked ? 'hi' : 'en';
        formData.append('lang', language);

        try {
            // Trigger API request
            const response = await fetch('/api/caption', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Failed to process request.' }));
                throw new Error(errorData.detail || `Server returned status code ${response.status}`);
            }

            const data = await response.json();
            displayResults(data);

        } catch (err) {
            console.error(err);
            showError(err.message || 'An unexpected error occurred. Please try again.');
        } finally {
            showLoading(false);
        }
    });

    // --- Display Results ---
    function displayResults(data) {
        // Load image preview via FileReader
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreview.alt = `Uploaded image. AI English description is: ${data.caption_en}`;
        };
        reader.readAsDataURL(selectedFile);

        // Populate captions
        textCaptionEn.textContent = data.caption_en;
        
        if (data.caption_translated) {
            captionTranslatedBlock.classList.remove('hidden');
            textCaptionTranslated.textContent = data.caption_translated;
            // Focus on Hindi caption if translated so screen readers read it first
            setTimeout(() => textCaptionTranslated.focus(), 100);
        } else {
            captionTranslatedBlock.classList.add('hidden');
            textCaptionTranslated.textContent = '';
            setTimeout(() => textCaptionEn.focus(), 100);
        }

        // Setup audio
        const audioSrc = `data:audio/mp3;base64,${data.audio_base64}`;
        audioPlayer.src = audioSrc;
        audioPlayer.load();

        // Screen readers announce when audio is loaded and ready
        audioPlayer.setAttribute('aria-label', `Spoken image caption narration: ${data.caption_translated || data.caption_en}`);

        // Setup download button
        btnDownload.href = audioSrc;
        btnDownload.download = `caption_${selectedFile.name.split('.')[0]}.mp3`;

        // Show result section
        resultsSection.classList.remove('hidden');
        
        // Auto play narration for accessibility
        audioPlayer.play().catch(e => {
            // Browsers often block autoplay without user interaction, ignore safely
            console.log("Autoplay prevented by browser security policy. User needs to play manually.");
        });

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // --- Helpers ---
    function showLoading(isLoading) {
        if (isLoading) {
            loadingSpinner.classList.remove('hidden');
            btnGenerate.disabled = true;
            btnGenerate.textContent = 'Processing...';
            // Alert screen readers
            document.getElementById('loading-text').setAttribute('aria-live', 'assertive');
        } else {
            loadingSpinner.classList.add('hidden');
            btnGenerate.disabled = false;
            btnGenerate.textContent = 'Generate Caption & Audio';
        }
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('hidden');
        errorMessage.scrollIntoView({ behavior: 'smooth' });
    }

    function hideError() {
        errorMessage.textContent = '';
        errorMessage.classList.add('hidden');
    }

    function hideResults() {
        resultsSection.classList.add('hidden');
        imagePreview.src = '';
        textCaptionEn.textContent = '';
        textCaptionTranslated.textContent = '';
        audioPlayer.src = '';
        btnDownload.href = '#';
    }
});
