import React, { useEffect, useRef, useState, useCallback } from "react";
import { Mic, MicOff, Play, Square, RotateCcw, Volume2, Loader, 
         AlertCircle, CheckCircle, Info, Zap } from "lucide-react";
import Header from "../components/Header";
import api from "../api";

const AudioRecognizer = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [hasRecording, setHasRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [recordingStatus, setRecordingStatus] = useState('idle');

  const timerRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioRef = useRef(null);

  const formatTime = useCallback((s) => {
    const min = String(Math.floor(s / 60)).padStart(2, "0");
    const sec = String(s % 60).padStart(2, "0");
    return `${min}:${sec}`;
  }, []);

  const startTimer = useCallback(() => {
    if (timerRef.current) return;
    timerRef.current = setInterval(() => {
      setSeconds((prev) => prev + 1);
    }, 1000);
  }, []);

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const resetRecording = useCallback(() => {
    setSeconds(0);
    setHasRecording(false);
    setAudioBlob(null);
    setRecordingStatus('idle');
    setIsPlaying(false);
    audioChunksRef.current = [];
  }, []);

  const startRecording = useCallback(async () => {
    try { 
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setAudioBlob(audioBlob);
        setHasRecording(true);
        setRecordingStatus('completed');
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingStatus('recording');
      startTimer();
    } catch (error) {
      console.error('Error accessing microphone:', error);
      setError(`Помилка доступу до мікрофона: ${error.message}`);
    }
  }, [startTimer]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      stopTimer();
    }
  }, [isRecording, stopTimer]);

  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      if (hasRecording) {
        resetRecording();
      }
      startRecording();
    }
  }, [isRecording, hasRecording, stopRecording, startRecording, resetRecording]);

  const playRecording = useCallback(() => {
    if (audioBlob && audioRef.current) {
      const audioUrl = URL.createObjectURL(audioBlob);
      audioRef.current.src = audioUrl;
      audioRef.current.play();
      setIsPlaying(true);

      audioRef.current.onended = () => {
        setIsPlaying(false);
        URL.revokeObjectURL(audioUrl);
      };
    }
  }, [audioBlob]);

  const audioToBase64 = useCallback((blob) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const arrayBuffer = reader.result;
        const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsArrayBuffer(blob);
    });
  }, []);

  const analyzeAudio = useCallback(async () => {
    if (!audioBlob) return;
    
    setError(null);

    try {
      const audioBase64 = await audioToBase64(audioBlob);
      
      const requestData = {
        audio_data: audioBase64,
        format: 'webm'
      };

      const response = await api.post(endpoint, requestData);
      

    } catch (error) {
      console.error('Error analyzing audio:', error);
      setError(error.response?.data?.error || 'Помилка аналізу аудіо');
    } finally {
    }
  }, [audioBlob, audioToBase64]);

  useEffect(() => {
    return () => {
      stopTimer();
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop();
      }
    };
  }, [stopTimer]);

  const getRecordingButtonClass = () => {
    const baseClass = "record-btn";
    if (isRecording) return `${baseClass} ${baseClass}--recording`;
    if (hasRecording) return `${baseClass} ${baseClass}--completed`;
    return baseClass;
  };

  return (
    <>
      <Header />
      <div className="audio-recognizer">
        <div className="audio-recognizer__container">
          <div className="audio-recognizer__header">
            <h1 className="audio-recognizer__title">Розпізнавання музичних інтервалів</h1>
            <p className="audio-recognizer__subtitle">
              Запишіть звук та отримайте передбачення музичного інтервалу
            </p>
          </div>

          <div className="audio-recognizer__content">
            <div className="audio-recognizer__recorder">
              <div className="audio-recognizer__timer">
                <div className="timer-display">
                  <span className="timer-display__time">{formatTime(seconds)}</span>
                  {isRecording && (
                    <div className="timer-display__indicator">
                      <div className="recording-pulse"></div>
                    </div>
                  )}
                </div>
              </div>

              <div className="audio-recognizer__controls">
                <button
                  className={getRecordingButtonClass()}
                  onClick={toggleRecording}
                  aria-label={isRecording ? "Зупинити запис" : "Почати запис"}
                >
                  <div className="record-btn__icon">
                    {isRecording ? <Square /> : <Mic />}
                  </div>
                  <span className="record-btn__text">
                    {isRecording ? "Зупинити запис" : hasRecording ? "Новий запис" : "Почати запис"}
                  </span>
                </button>

                {hasRecording && (
                  <div className="audio-recognizer__actions">
                    <button
                      className="action-btn action-btn--secondary"
                      onClick={playRecording}
                      disabled={isPlaying}
                      aria-label="Відтворити запис"
                    >
                      <Play />
                      <span>{isPlaying ? "Відтворюється..." : "Прослухати"}</span>
                    </button>

                    <button
                      className="action-btn action-btn--primary"
                      onClick={analyzeAudio}
                      aria-label="Аналізувати запис"
                    >
                      <Volume2 />
                      <span>Аналізувати</span>
                    </button>

                    <button
                      className="action-btn action-btn--ghost"
                      onClick={resetRecording}
                      aria-label="Скинути запис"
                    >
                      <RotateCcw />
                      <span>Скинути</span>
                    </button>
                  </div>
                )}
              </div>
            </div>

            { (
              <div className="results-panel">
                <h2 className="results-panel__title">Результат розпізнавання</h2>
                
                { (
                  <div className="recognition-success">
                    <div className="main-result">
                      <div className="interval-display">
                        <div className="interval-display__name">
                        </div>
                        <div className="interval-display__confidence">
                          Впевненість: %
                        </div>
                      </div>
                    </div>

                    { (
                      <div className="alternative-results">
                        <h3>Альтернативні варіанти:</h3>
                        <div className="alternatives-list">
                          
                        </div>
                      </div>
                    )}

                    { (
                      <div className="recommendations">
                        <h3>Рекомендації:</h3>
                        <ul className="recommendations-list">
                          
                        </ul>
                      </div>
                    )}

                    { (
                      <div className="processing-info">
                        <div className="info-grid">
                          <div className="info-item">
                            <span className="info-label">Сегментів проаналізовано:</span>
                            <span className="info-value"></span>
                          </div>
                          <div className="info-item">
                            <span className="info-label">Час обробки:</span>
                            <span className="info-value">
                              
                            </span>
                          </div>
                          <div className="info-item">
                            <span className="info-label">Якість запису:</span>
                            <span className="info-value">
                              %
                            </span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) }
              </div>
            )}

            {recordingStatus === 'completed' && (
              <div className="audio-recognizer__status">
                <div className="status-card status-card--success">
                  <div className="status-card__icon">
                    <Volume2 />
                  </div>
                  <div className="status-card__content">
                    <h3 className="status-card__title">Запис завершено</h3>
                    <p className="status-card__description">
                      Тривалість: {formatTime(seconds)}. Натисніть "Аналізувати" для розпізнавання інтервалу.
                    </p>
                  </div>
                </div>
              </div>
            )}

            { (
              <div className="audio-recognizer__status">
                <div className="status-card status-card--error">
                  <div className="status-card__icon">
                    <AlertCircle />
                  </div>
                  <div className="status-card__content">
                    <h3 className="status-card__title">Помилка</h3>
                    <p className="status-card__description"></p>
                  </div>
                </div>
              </div>
            )}
          </div>

          <audio ref={audioRef} style={{ display: 'none' }} />
        </div>
      </div>
    </>
  );
};

export default AudioRecognizer;