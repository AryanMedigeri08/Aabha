"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Upload, Download, AlertCircle, Volume2 } from 'lucide-react';

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
        // A short rising chime (pleasant upload feedback)
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        
        osc.type = 'sine';
        osc.frequency.setValueAtTime(440, ctx.currentTime); // A4
        osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.15); // A5
        
        gain.gain.setValueAtTime(0.15, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.15);
        
        osc.start();
        osc.stop(ctx.currentTime + 0.15);
      } else if (type === 'processing') {
        // Low soft pulse indicating background work
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(220, ctx.currentTime); // A3
        
        gain.gain.setValueAtTime(0.05, ctx.currentTime);
        gain.gain.linearRampToValueAtTime(0.05, ctx.currentTime + 0.2);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.4);
        
        osc.start();
        osc.stop(ctx.currentTime + 0.4);
      } else if (type === 'success') {
        // Warm, harmonic completion tone (two-tone chord)
        const now = ctx.currentTime;
        [523.25, 659.25].forEach((freq) => { // C5 and E5 chord
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
        // Low error warning tone (A2/G#2 double beep)
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

      // Shortcut: U or Alt+U - Trigger file uploader click
      if (e.key.toLowerCase() === 'u' || (e.altKey && e.key.toLowerCase() === 'u')) {
        e.preventDefault();
        fileInputRef.current?.click();
      }

      // Shortcut: R or Alt+R - Replay main narration audio
      if (e.key.toLowerCase() === 'r' || (e.altKey && e.key.toLowerCase() === 'r')) {
        e.preventDefault();
        if (audioPlayerRef.current) {
          audioPlayerRef.current.currentTime = 0;
          audioPlayerRef.current.playbackRate = playbackSpeed;
          audioPlayerRef.current.play().catch(err => console.log(err));
        }
      }

      // Shortcut: L or Alt+L - Toggle language
      if (e.key.toLowerCase() === 'l' || (e.altKey && e.key.toLowerCase() === 'l')) {
        e.preventDefault();
        setLanguage((prev) => (prev === 'en' ? 'hi' : 'en'));
      }

      // Shortcut: S or Alt+S - Focus chat input
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

  // Handle file input changes
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  // Keyboard navigation trigger for upload zone
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInputRef.current.click();
    }
  };

  // Process and validate selected file
  const processFile = (file) => {
    setError(null);
    setResults(null);

    // Validate type
    if (!ALLOWED_TYPES.includes(file.type)) {
      setError(`Unsupported file format (${file.type || 'unknown'}). Please upload a JPEG, PNG, or WEBP image.`);
      setSelectedFile(null);
      setImagePreviewUrl(null);
      return;
    }

    // Validate size
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

  // Drag and drop event handlers
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

  // Trigger caption generation
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
      // Next.js config proxy forwards this rewrite request to http://localhost:8000/api/caption
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

      // Scroll to results once state is set
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth' });
        captionRef.current?.focus();
        
        // Attempt autoplay for screen readers / accessibility narration
        if (audioPlayerRef.current) {
          audioPlayerRef.current.playbackRate = playbackSpeed;
          audioPlayerRef.current.play().catch(() => {
            console.log("Autoplay blocked by browser. User interaction required.");
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
      const audioSrc = `data:audio/mp3;base64,${data.audio_base64}`;
      
      setChatHistory([...newHistory, { sender: 'ai', text: answer, audioSrc }]);
      setChatAudioSrc(audioSrc);
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
      <header className="app-header">
        <div class="header-container">
          <h1 id="app-title" className="brand-title">Aabha</h1>
          <p className="brand-subtitle">Empowering visually impaired users through intelligent image description & audio narration.</p>
        </div>
      </header>

      <main className="main-container">
        {/* Main interactive panel */}
        <section className="card glass-panel" aria-labelledby="uploader-heading">
          <h2 id="uploader-heading" className="sr-only">Image Uploader and Settings</h2>
          
          {/* Upload Drop Zone */}
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
            <div className="upload-icon" aria-hidden="true">
              <Upload size={64} />
            </div>
            
            <div id="upload-instructions" className="upload-text">
              {selectedFile ? (
                <>Selected file: <span className="highlight">{selectedFile.name}</span></>
              ) : (
                <><span className="highlight">Drag & drop your image here</span> or <span className="browse-link">browse files</span></>
              )}
            </div>
            
            <div className="upload-info">Supports JPEG, PNG, WEBP (Max 10MB)</div>
            
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

          {/* Configuration Panel */}
          <div className="controls-panel">
            <fieldset className="language-selector">
              <legend className="section-title">Select Caption Language / भाषा चुनें</legend>
              <div className="radio-group">
                <label className="radio-label" htmlFor="lang-en">
                  <input 
                    type="radio" 
                    id="lang-en" 
                    name="language" 
                    value="en" 
                    checked={language === 'en'}
                    onChange={handleRadioChange}
                  />
                  <span className="custom-radio"></span>
                  <span className="label-text">English</span>
                </label>
                
                <label className="radio-label" htmlFor="lang-hi">
                  <input 
                    type="radio" 
                    id="lang-hi" 
                    name="language" 
                    value="hi" 
                    checked={language === 'hi'}
                    onChange={handleRadioChange}
                  />
                  <span className="custom-radio"></span>
                  <span className="label-text">Hindi / हिंदी</span>
                </label>
              </div>
            </fieldset>
          </div>

          {/* Error Message Box */}
          {error && (
            <div id="error-message" className="error-container" role="alert" aria-live="assertive">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
            </div>
          )}

          {/* Generate Button Action */}
          <div className="action-bar">
            <button 
              type="button" 
              id="btn-generate" 
              className="btn-primary" 
              disabled={!selectedFile || loading}
              onClick={handleGenerate}
              aria-describedby={!selectedFile ? "btn-generate-desc" : undefined}
            >
              {loading ? 'Processing...' : 'Generate Caption & Audio'}
            </button>
            {!selectedFile && (
              <div id="btn-generate-desc" className="sr-only">Upload an image to enable this button and generate narration.</div>
            )}
          </div>
        </section>

        {/* Loading Spinner */}
        {loading && (
          <div id="loading-spinner" className="spinner-container" role="status" aria-live="polite">
            <div className="spinner"></div>
            <p id="loading-text" className="loading-message">Analyzing image content...</p>
          </div>
        )}

        {/* Results Output Block */}
        {results && (
          <section id="results-section" ref={resultsRef} className="card glass-panel results-fade-in" aria-labelledby="results-heading">
            <h2 id="results-heading" className="section-title">Narration Results</h2>
            
            <div className="results-grid">
              {/* Image Preview Container */}
              <div className="preview-panel">
                <h3 className="panel-subtitle">Uploaded Image Preview</h3>
                <div className="image-preview-container">
                  {imagePreviewUrl && (
                    <img 
                      id="image-preview" 
                      src={imagePreviewUrl} 
                      alt={`Uploaded image. AI generated caption: ${results.caption_en}`} 
                      className="image-preview" 
                    />
                  )}
                </div>
              </div>

              {/* Text Output and Audio Player */}
              <div className="output-panel">
                <div className="caption-container">
                  <h3 class="panel-subtitle">Generated Description</h3>
                  
                  {/* English Caption */}
                  <div className="caption-block english-caption">
                    <div className="caption-header">
                      <span className="lang-tag">English</span>
                    </div>
                    <p 
                      id="text-caption-en" 
                      className="caption-text" 
                      tabIndex={0} 
                      ref={language === 'en' ? captionRef : null}
                      aria-label="English caption"
                    >
                      {results.caption_en}
                    </p>
                  </div>

                  {/* Hindi Translation */}
                  {results.caption_translated && (
                    <div id="caption-translated-block" className="caption-block hindi-caption">
                      <div className="caption-header">
                        <span className="lang-tag">Hindi / हिंदी</span>
                      </div>
                      <p 
                        id="text-caption-translated" 
                        className="caption-text" 
                        tabIndex={0} 
                        ref={language === 'hi' ? captionRef : null}
                        aria-label="Hindi translation"
                      >
                        {results.caption_translated}
                      </p>
                    </div>
                  )}
                </div>

                {/* Audio Narrator Component */}
                <div className="audio-container">
                  <h3 className="panel-subtitle">Audio Narration</h3>
                  <div className="audio-wrapper">
                    <audio 
                      id="audio-player" 
                      className="native-audio" 
                      controls 
                      src={audioSrc}
                      ref={audioPlayerRef}
                      onPlay={(e) => { e.target.playbackRate = playbackSpeed; }}
                      onCanPlay={(e) => { e.target.playbackRate = playbackSpeed; }}
                      aria-label={`Spoken image caption narration: ${results.caption_translated || results.caption_en}`}
                    />
                    
                    <div className="speed-control-container">
                      <label htmlFor="playback-speed-select" className="sr-only">Playback Speed</label>
                      <select 
                        id="playback-speed-select" 
                        className="speed-select" 
                        value={playbackSpeed}
                        onChange={handleSpeedChange}
                        aria-label="Select playback speed"
                      >
                        <option value="0.75">0.75x</option>
                        <option value="1.0">1.0x (Normal)</option>
                        <option value="1.25">1.25x</option>
                        <option value="1.5">1.5x</option>
                        <option value="2.0">2.0x</option>
                      </select>
                    </div>
                    
                    <a 
                      id="btn-download" 
                      href={audioSrc} 
                      download={`caption_${selectedFile?.name?.split('.')[0] || 'narration'}.mp3`} 
                      className="btn-secondary" 
                      aria-label="Download audio narration as MP3 file"
                    >
                      <Download size={20} className="btn-icon" aria-hidden="true" />
                      Download Audio
                    </a>
                  </div>
                </div>

                {/* VQA Chat Panel */}
                <div className="chat-interface-card">
                  <h3 className="panel-subtitle">Ask questions about this image / सवाल पूछें</h3>
                  
                  <div className="chat-messages-box" aria-live="polite">
                    {chatHistory.length === 0 ? (
                      <p className="chat-placeholder">Ask anything! e.g., "What is in the background?" or "What color is the car?"</p>
                    ) : (
                      chatHistory.map((msg, idx) => (
                        <div key={idx} className={`chat-message ${msg.sender}-message`}>
                          <div className="message-content">
                            <span className="message-sender-label">{msg.sender === 'user' ? 'You' : 'Aabha'}</span>
                            <p className="message-text">{msg.text}</p>
                          </div>
                          {msg.audioSrc && (
                            <button 
                              type="button" 
                              onClick={() => {
                                setChatAudioSrc(msg.audioSrc);
                                setTimeout(() => {
                                  if (chatAudioPlayerRef.current) {
                                    chatAudioPlayerRef.current.playbackRate = playbackSpeed;
                                    chatAudioPlayerRef.current.play().catch(err => console.log(err));
                                  }
                                }, 50);
                              }}
                              className="chat-audio-btn"
                              aria-label="Replay audio answer"
                            >
                              <Volume2 size={16} />
                            </button>
                          )}
                        </div>
                      ))
                    )}
                    <div ref={chatBottomRef} />
                  </div>
                  
                  <form onSubmit={handleChatSubmit} className="chat-input-form">
                    <input 
                      type="text" 
                      id="chat-question-input"
                      ref={chatInputRef}
                      className="chat-text-input" 
                      value={chatQuestion}
                      onChange={(e) => setChatQuestion(e.target.value)}
                      placeholder="Ask a question about the image..."
                      disabled={chatLoading}
                      aria-label="Type your question about the image"
                    />
                    <button 
                      type="submit" 
                      className="btn-chat-submit" 
                      disabled={!chatQuestion.trim() || chatLoading}
                    >
                      {chatLoading ? 'Thinking...' : 'Ask'}
                    </button>
                  </form>
                  
                  <audio 
                    ref={chatAudioPlayerRef} 
                    src={chatAudioSrc} 
                    style={{ display: 'none' }} 
                  />
                </div>
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="app-footer">
        <div className="footer-container">
          <p>&copy; 2026 Aabha. Designed with accessibility at the core.</p>
        </div>
      </footer>
    </>
  );
}
