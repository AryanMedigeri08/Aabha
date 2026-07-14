"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Download, AlertCircle, Volume2 } from 'lucide-react';

/* ============================================
   Inline SVG Components — Accessibility Themed
   ============================================ */

// Hero: Stylized eye with radiating sound waves — "seeing through sound"
const EyeSoundWaveSvg = () => (
  <svg className="hero-svg" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    {/* Sound wave arcs */}
    <path className="sound-wave" d="M90 60c0-16.57-13.43-30-30-30" stroke="#6C3CE1" strokeWidth="2" strokeLinecap="round" fill="none" opacity="0.3"/>
    <path className="sound-wave" d="M97 60c0-20.43-16.57-37-37-37" stroke="#6C3CE1" strokeWidth="1.5" strokeLinecap="round" fill="none" opacity="0.2"/>
    <path className="sound-wave" d="M104 60c0-24.3-19.7-44-44-44" stroke="#6C3CE1" strokeWidth="1" strokeLinecap="round" fill="none" opacity="0.15"/>
    {/* Mirror waves */}
    <path className="sound-wave" d="M30 60c0 16.57 13.43 30 30 30" stroke="#E8734A" strokeWidth="2" strokeLinecap="round" fill="none" opacity="0.3"/>
    <path className="sound-wave" d="M23 60c0 20.43 16.57 37 37 37" stroke="#E8734A" strokeWidth="1.5" strokeLinecap="round" fill="none" opacity="0.2"/>
    <path className="sound-wave" d="M16 60c0 24.3 19.7 44 44 44" stroke="#E8734A" strokeWidth="1" strokeLinecap="round" fill="none" opacity="0.15"/>
    {/* Eye shape */}
    <path d="M60 38C45 38 33 50 28 60c5 10 17 22 32 22s27-12 32-22c-5-10-17-22-32-22z" fill="#F3F0FF" stroke="#6C3CE1" strokeWidth="2.5"/>
    {/* Iris */}
    <circle cx="60" cy="60" r="12" fill="#6C3CE1" opacity="0.9"/>
    {/* Pupil */}
    <circle cx="60" cy="60" r="5" fill="#1A1A2E"/>
    {/* Light reflection */}
    <circle cx="56" cy="56" r="2.5" fill="white" opacity="0.8"/>
  </svg>
);

// Upload illustration: Hand reaching toward an image frame with ripples
const HandImageSvg = () => (
  <svg className="upload-svg" viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    {/* Image frame */}
    <rect x="20" y="16" width="40" height="32" rx="4" fill="#F3F0FF" stroke="#6C3CE1" strokeWidth="1.5"/>
    {/* Mountain scenery inside frame */}
    <path d="M24 44l8-10 6 6 8-12 10 16H24z" fill="#E8734A" opacity="0.2"/>
    <circle cx="34" cy="26" r="3" fill="#E8734A" opacity="0.3"/>
    {/* Touch ripples */}
    <circle cx="40" cy="60" r="6" stroke="#6C3CE1" strokeWidth="1" fill="none" opacity="0.3">
      <animate attributeName="r" values="6;12" dur="2s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.3;0" dur="2s" repeatCount="indefinite"/>
    </circle>
    <circle cx="40" cy="60" r="6" stroke="#6C3CE1" strokeWidth="1" fill="none" opacity="0.3">
      <animate attributeName="r" values="6;16" dur="2s" begin="0.5s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.2;0" dur="2s" begin="0.5s" repeatCount="indefinite"/>
    </circle>
    {/* Fingertip */}
    <ellipse cx="40" cy="60" rx="4" ry="5" fill="#6C3CE1" opacity="0.15"/>
    <circle cx="40" cy="58" r="2.5" fill="#6C3CE1" opacity="0.4"/>
    {/* Connecting line */}
    <line x1="40" y1="48" x2="40" y2="55" stroke="#6C3CE1" strokeWidth="1" strokeDasharray="2 2" opacity="0.3"/>
  </svg>
);

// Braille loading dots (6-dot cell pattern)
const BrailleLoader = () => (
  <div className="braille-loader" aria-hidden="true">
    <div className="braille-dot"></div>
    <div className="braille-dot"></div>
    <div className="braille-dot"></div>
    <div className="braille-dot"></div>
    <div className="braille-dot"></div>
    <div className="braille-dot"></div>
  </div>
);

// Audio waveform visualization
const AudioWaveformSvg = () => (
  <svg width="48" height="24" viewBox="0 0 48 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" style={{ display: 'inline-block', verticalAlign: 'middle', marginRight: '0.5rem' }}>
    {[4, 12, 20, 28, 36, 44].map((x, i) => (
      <rect key={i} x={x - 1.5} y={8 - i % 3 * 2} width="3" rx="1.5" fill="#6C3CE1" opacity="0.5">
        <animate attributeName="height" values={`${8 + i * 2};${14 + (5 - i) * 2};${8 + i * 2}`} dur={`${1 + i * 0.15}s`} repeatCount="indefinite"/>
        <animate attributeName="y" values={`${8 - i % 3 * 2};${5 - (5 - i) % 3};${8 - i % 3 * 2}`} dur={`${1 + i * 0.15}s`} repeatCount="indefinite"/>
      </rect>
    ))}
  </svg>
);


/* ============================================
   Main Page Component
   ============================================ */
export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [language, setLanguage] = useState('en');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [chatHistory, setChatHistory] = useState([]);
  const [chatQuestion, setChatQuestion] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatAudioSrc, setChatAudioSrc] = useState("");

  const fileInputRef = useRef(null);
  const audioPlayerRef = useRef(null);
  const resultsRef = useRef(null);
  const captionRef = useRef(null);
  const chatAudioPlayerRef = useRef(null);
  const chatBottomRef = useRef(null);
  const chatInputRef = useRef(null);

  const MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10MB
  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

  // Web Audio API Sound Effects for Accessibility
  const playBeep = (type) => {
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) return;
      const ctx = new AudioContext();

      if (type === 'upload') {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.type = 'sine';
        osc.frequency.setValueAtTime(440, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.15);
        gain.gain.setValueAtTime(0.15, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.15);
        osc.start();
        osc.stop(ctx.currentTime + 0.15);
      } else if (type === 'processing') {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(220, ctx.currentTime);
        gain.gain.setValueAtTime(0.05, ctx.currentTime);
        gain.gain.linearRampToValueAtTime(0.05, ctx.currentTime + 0.2);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.4);
        osc.start();
        osc.stop(ctx.currentTime + 0.4);
      } else if (type === 'success') {
        const now = ctx.currentTime;
        [523.25, 659.25].forEach((freq) => {
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.type = 'sine';
          osc.frequency.setValueAtTime(freq, now);
          gain.gain.setValueAtTime(0.1, now);
          gain.gain.exponentialRampToValueAtTime(0.01, now + 0.3);
          osc.start();
          osc.stop(now + 0.3);
        });
      } else if (type === 'error') {
        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(150, now);
        gain.gain.setValueAtTime(0.1, now);
        gain.gain.linearRampToValueAtTime(0.1, now + 0.1);
        gain.gain.setValueAtTime(0.0, now + 0.15);
        gain.gain.setValueAtTime(0.1, now + 0.2);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.35);
        osc.start();
        osc.stop(now + 0.35);
      }
    } catch (e) {
      console.log("AudioContext failed or blocked:", e);
    }
  };

  // Clean up object URL to prevent memory leaks
  useEffect(() => {
    return () => {
      if (imagePreviewUrl) {
        URL.revokeObjectURL(imagePreviewUrl);
      }
    };
  }, [imagePreviewUrl]);

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleGlobalKeyDown = (e) => {
      const activeTag = document.activeElement?.tagName?.toLowerCase();
      if (activeTag === 'input' || activeTag === 'textarea' || activeTag === 'select') {
        if (e.key === 'Escape') {
          document.activeElement.blur();
        }
        return;
      }

      if (e.key.toLowerCase() === 'u' || (e.altKey && e.key.toLowerCase() === 'u')) {
        e.preventDefault();
        fileInputRef.current?.click();
      }

      if (e.key.toLowerCase() === 'r' || (e.altKey && e.key.toLowerCase() === 'r')) {
        e.preventDefault();
        if (audioPlayerRef.current) {
          audioPlayerRef.current.currentTime = 0;
          audioPlayerRef.current.playbackRate = playbackSpeed;
          audioPlayerRef.current.play().catch(err => console.log(err));
        }
      }

      if (e.key.toLowerCase() === 'l' || (e.altKey && e.key.toLowerCase() === 'l')) {
        e.preventDefault();
        setLanguage((prev) => (prev === 'en' ? 'hi' : 'en'));
      }

      if (e.key.toLowerCase() === 's' || (e.altKey && e.key.toLowerCase() === 's')) {
        e.preventDefault();
        chatInputRef.current?.focus();
      }
    };

    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => {
      window.removeEventListener('keydown', handleGlobalKeyDown);
    };
  }, [playbackSpeed, chatHistory]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInputRef.current.click();
    }
  };

  const processFile = (file) => {
    setError(null);
    setResults(null);

    if (!ALLOWED_TYPES.includes(file.type)) {
      setError(`Unsupported file format (${file.type || 'unknown'}). Please upload a JPEG, PNG, or WEBP image.`);
      setSelectedFile(null);
      setImagePreviewUrl(null);
      return;
    }

    if (file.size > MAX_SIZE_BYTES) {
      const sizeInMB = (file.size / (1024 * 1024)).toFixed(1);
      setError(`The file is too large (${sizeInMB}MB). Maximum allowed size is 10MB.`);
      setSelectedFile(null);
      setImagePreviewUrl(null);
      return;
    }

    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setImagePreviewUrl(objectUrl);
    setChatHistory([]);
    setChatAudioSrc("");
    playBeep('upload');
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleGenerate = async () => {
    if (!selectedFile) return;

    setError(null);
    setResults(null);
    setLoading(true);
    playBeep('processing');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('lang', language);

    try {
      const response = await fetch('/api/caption', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to process request.' }));
        throw new Error(errorData.detail || `Server returned status code ${response.status}`);
      }

      const data = await response.json();
      setResults(data);
      playBeep('success');

      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth' });
        captionRef.current?.focus();
        if (audioPlayerRef.current) {
          audioPlayerRef.current.playbackRate = playbackSpeed;
          audioPlayerRef.current.play().catch(() => {
            console.log("Autoplay blocked by browser.");
          });
        }
      }, 100);

    } catch (err) {
      console.error(err);
      setError(err.message || 'An unexpected error occurred. Please try again.');
      playBeep('error');
    } finally {
      setLoading(false);
    }
  };

  const handleRadioChange = (e) => {
    setLanguage(e.target.value);
  };

  const handleSpeedChange = (e) => {
    const speed = parseFloat(e.target.value);
    setPlaybackSpeed(speed);
    if (audioPlayerRef.current) {
      audioPlayerRef.current.playbackRate = speed;
    }
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatQuestion.trim() || !selectedFile || chatLoading) return;

    const currentQuestion = chatQuestion.trim();
    setChatQuestion("");
    setChatLoading(true);
    setError(null);
    playBeep('processing');

    const newHistory = [...chatHistory, { sender: 'user', text: currentQuestion }];
    setChatHistory(newHistory);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('question', currentQuestion);
    formData.append('lang', language);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to process chat question.' }));
        throw new Error(errorData.detail || `Server returned status code ${response.status}`);
      }

      const data = await response.json();
      const answer = data.answer_translated || data.answer_en;
      const chatAudio = `data:audio/mp3;base64,${data.audio_base64}`;
      
      setChatHistory([...newHistory, { sender: 'ai', text: answer, audioSrc: chatAudio }]);
      setChatAudioSrc(chatAudio);
      playBeep('success');

      setTimeout(() => {
        if (chatAudioPlayerRef.current) {
          chatAudioPlayerRef.current.playbackRate = playbackSpeed;
          chatAudioPlayerRef.current.play().catch(() => {
            console.log("Autoplay blocked for chat narration.");
          });
        }
        chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);

    } catch (err) {
      console.error(err);
      setError(err.message || 'Failed to get answer. Please try again.');
      playBeep('error');
    } finally {
      setChatLoading(false);
    }
  };

  const audioSrc = results ? `data:audio/mp3;base64,${results.audio_base64}` : '';

  return (
    <>
      {/* ===== Header with Hero Illustration ===== */}
      <header className="app-header">
        <div className="header-container">
          <div className="hero-illustration">
            <EyeSoundWaveSvg />
          </div>
          <h1 id="app-title" className="brand-title">Aabha</h1>
          <p className="brand-subtitle">
            Transforming images into spoken narratives — empowering visually impaired users to perceive the world through sound.
          </p>
          <div className="shortcuts-hint">
            <span className="kbd">U Upload</span>
            <span className="kbd">R Replay</span>
            <span className="kbd">L Language</span>
            <span className="kbd">S Chat</span>
          </div>
        </div>
      </header>

      <main className="main-container">
        {/* ===== Upload & Settings Card ===== */}
        <section className="card" aria-labelledby="uploader-heading">
          <h2 id="uploader-heading" className="sr-only">Image Uploader and Settings</h2>
          
          <div 
            id="drop-zone" 
            className={`upload-area ${dragActive ? 'drag-over' : ''}`}
            tabIndex={0} 
            role="button" 
            aria-controls="file-input" 
            aria-describedby="upload-instructions"
            onKeyDown={handleKeyDown}
            onClick={() => fileInputRef.current.click()}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <div className="upload-illustration">
              <HandImageSvg />
            </div>
            
            <div id="upload-instructions" className="upload-text">
              {selectedFile ? (
                <>Selected: <span className="highlight">{selectedFile.name}</span></>
              ) : (
                <><span className="highlight">Drag & drop your image here</span> or <span className="browse-link">browse files</span></>
              )}
            </div>
            
            <div className="upload-info">Supports JPEG, PNG, WEBP · Max 10MB</div>
            
            <input 
              type="file" 
              id="file-input" 
              className="file-input" 
              accept=".jpg,.jpeg,.png,.webp" 
              ref={fileInputRef}
              onChange={handleFileChange}
              aria-label="Upload image file"
            />
          </div>

          {/* Language Selector */}
          <div className="controls-panel">
            <fieldset className="language-selector">
              <legend className="section-title">Caption Language / भाषा चुनें</legend>
              <div className="radio-group">
                <label className="radio-label" htmlFor="lang-en">
                  <input 
                    type="radio" id="lang-en" name="language" value="en" 
                    checked={language === 'en'} onChange={handleRadioChange}
                  />
                  <span className="custom-radio"></span>
                  <span className="label-text">English</span>
                </label>
                <label className="radio-label" htmlFor="lang-hi">
                  <input 
                    type="radio" id="lang-hi" name="language" value="hi" 
                    checked={language === 'hi'} onChange={handleRadioChange}
                  />
                  <span className="custom-radio"></span>
                  <span className="label-text">Hindi / हिंदी</span>
                </label>
              </div>
            </fieldset>
          </div>

          {/* Error */}
          {error && (
            <div id="error-message" className="error-container" role="alert" aria-live="assertive">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <AlertCircle size={18} />
                <span>{error}</span>
              </div>
            </div>
          )}

          {/* Generate Button */}
          <div className="action-bar">
            <button 
              type="button" id="btn-generate" className="btn-primary" 
              disabled={!selectedFile || loading} onClick={handleGenerate}
              aria-describedby={!selectedFile ? "btn-generate-desc" : undefined}
            >
              {loading ? 'Analyzing...' : 'Generate Caption & Audio'}
            </button>
            {!selectedFile && (
              <div id="btn-generate-desc" className="sr-only">Upload an image to enable this button.</div>
            )}
          </div>
        </section>

        {/* ===== Loading — Braille Dots ===== */}
        {loading && (
          <div id="loading-spinner" className="spinner-container" role="status" aria-live="polite">
            <BrailleLoader />
            <p id="loading-text" className="loading-message">Reading your image...</p>
          </div>
        )}

        {/* ===== Results ===== */}
        {results && (
          <section id="results-section" ref={resultsRef} className="card results-fade-in" aria-labelledby="results-heading">
            <h2 id="results-heading" className="section-title">
              <AudioWaveformSvg />
              Narration Results
            </h2>
            
            <div className="results-grid">
              {/* Image Preview */}
              <div className="preview-panel">
                <h3 className="panel-subtitle">Uploaded Image</h3>
                <div className="image-preview-container">
                  {imagePreviewUrl && (
                    <img 
                      id="image-preview" src={imagePreviewUrl} 
                      alt={`Uploaded image. AI generated caption: ${results.caption_en}`} 
                      className="image-preview" 
                    />
                  )}
                </div>
              </div>

              {/* Captions & Audio */}
              <div className="output-panel">
                <div className="caption-container">
                  <h3 className="panel-subtitle">Generated Description</h3>
                  
                  <div className="caption-block english-caption">
                    <div className="caption-header">
                      <span className="lang-tag">English</span>
                    </div>
                    <p 
                      id="text-caption-en" className="caption-text" tabIndex={0} 
                      ref={language === 'en' ? captionRef : null} aria-label="English caption"
                    >
                      {results.caption_en}
                    </p>
                  </div>

                  {results.caption_translated && (
                    <div id="caption-translated-block" className="caption-block hindi-caption">
                      <div className="caption-header">
                        <span className="lang-tag">Hindi / हिंदी</span>
                      </div>
                      <p 
                        id="text-caption-translated" className="caption-text" tabIndex={0} 
                        ref={language === 'hi' ? captionRef : null} aria-label="Hindi translation"
                      >
                        {results.caption_translated}
                      </p>
                    </div>
                  )}
                </div>

                {/* Audio Player */}
                <div className="audio-container">
                  <h3 className="panel-subtitle">Audio Narration</h3>
                  <div className="audio-wrapper">
                    <audio 
                      id="audio-player" className="native-audio" controls src={audioSrc}
                      ref={audioPlayerRef}
                      onPlay={(e) => { e.target.playbackRate = playbackSpeed; }}
                      onCanPlay={(e) => { e.target.playbackRate = playbackSpeed; }}
                      aria-label={`Spoken narration: ${results.caption_translated || results.caption_en}`}
                    />
                    
                    <div className="speed-control-container">
                      <label htmlFor="playback-speed-select" className="sr-only">Playback Speed</label>
                      <select 
                        id="playback-speed-select" className="speed-select" 
                        value={playbackSpeed} onChange={handleSpeedChange}
                        aria-label="Select playback speed"
                      >
                        <option value="0.75">0.75×</option>
                        <option value="1.0">1.0× Normal</option>
                        <option value="1.25">1.25×</option>
                        <option value="1.5">1.5×</option>
                        <option value="2.0">2.0×</option>
                      </select>
                    </div>
                    
                    <a 
                      id="btn-download" href={audioSrc} className="btn-secondary"
                      download={`caption_${selectedFile?.name?.split('.')[0] || 'narration'}.mp3`} 
                      aria-label="Download audio narration as MP3 file"
                    >
                      <Download size={18} className="btn-icon" aria-hidden="true" />
                      Download
                    </a>
                  </div>
                </div>

                {/* VQA Chat */}
                <div className="chat-interface-card">
                  <h3 className="panel-subtitle">Ask about this image / सवाल पूछें</h3>
                  
                  <div className="chat-messages-box" aria-live="polite">
                    {chatHistory.length === 0 ? (
                      <p className="chat-placeholder">Ask anything — e.g., &quot;What color is the car?&quot; or &quot;How many people?&quot;</p>
                    ) : (
                      chatHistory.map((msg, idx) => (
                        <div key={idx} className={`chat-message ${msg.sender}-message`}>
                          <div className="message-content">
                            <span className="message-sender-label">{msg.sender === 'user' ? 'You' : 'Aabha'}</span>
                            <p className="message-text">{msg.text}</p>
                          </div>
                          {msg.audioSrc && (
                            <button 
                              type="button" className="chat-audio-btn" aria-label="Replay audio answer"
                              onClick={() => {
                                setChatAudioSrc(msg.audioSrc);
                                setTimeout(() => {
                                  if (chatAudioPlayerRef.current) {
                                    chatAudioPlayerRef.current.playbackRate = playbackSpeed;
                                    chatAudioPlayerRef.current.play().catch(err => console.log(err));
                                  }
                                }, 50);
                              }}
                            >
                              <Volume2 size={14} />
                            </button>
                          )}
                        </div>
                      ))
                    )}
                    <div ref={chatBottomRef} />
                  </div>
                  
                  <form onSubmit={handleChatSubmit} className="chat-input-form">
                    <input 
                      type="text" id="chat-question-input" ref={chatInputRef}
                      className="chat-text-input" value={chatQuestion}
                      onChange={(e) => setChatQuestion(e.target.value)}
                      placeholder="Ask a question about the image..."
                      disabled={chatLoading} aria-label="Type your question about the image"
                    />
                    <button 
                      type="submit" className="btn-chat-submit" 
                      disabled={!chatQuestion.trim() || chatLoading}
                    >
                      {chatLoading ? 'Thinking...' : 'Ask'}
                    </button>
                  </form>
                  
                  <audio ref={chatAudioPlayerRef} src={chatAudioSrc} style={{ display: 'none' }} />
                </div>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* ===== Footer with Braille Decoration ===== */}
      <footer className="app-footer">
        <div className="footer-container">
          <div className="footer-braille-art" aria-hidden="true">
            {/* Braille pattern for "Aabha" — decorative only */}
            {[1,0,1,0,1,1,0,1,0,1,1,0].map((filled, i) => (
              <div key={i} className="dot" style={{ opacity: filled ? 1 : 0.2 }}></div>
            ))}
          </div>
          <p>&copy; 2026 Aabha. Designed with accessibility at the core.</p>
        </div>
      </footer>
    </>
  );
}
