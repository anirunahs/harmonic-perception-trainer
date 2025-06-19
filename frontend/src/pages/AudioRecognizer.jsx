import React, { useEffect, useRef, useState, useCallback } from "react";
import { Mic, MicOff, Play, Square, RotateCcw, Volume2, Loader, 
         AlertCircle, CheckCircle, Info, Zap, Settings, TrendingUp } from "lucide-react";
import Header from "../components/Header";
import RecognitionResults from "../components/recognition/RecognitionResults";
import api from "../api";

const AudioRecognizer = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [hasRecording, setHasRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [recordingStatus, setRecordingStatus] = useState('idle');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState(null);
  const [audioQuality, setAudioQuality] = useState(null);
  const [processingSettings, setProcessingSettings] = useState({
    preprocessing_level: 'standard',
    max_segments: 3,
    noise_reduction: true
  });

  const timerRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioRef = useRef(null);
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);

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
    setAnalysisResult(null);
    setError(null);
    setAudioQuality(null);
    audioChunksRef.current = [];
  }, []);

  const setupAudioContext = useCallback(async (stream) => {
    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 44100
      });
      
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.8;
      
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      
      audioContextRef.current = audioContext;
      analyserRef.current = analyser;
      
      return { audioContext, analyser };
    } catch (error) {
      console.error('Error setting up audio context:', error);
      return null;
    }
  }, []);

  const analyzeAudioQuality = useCallback(() => {
    if (!analyserRef.current) return null;

    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyserRef.current.getByteFrequencyData(dataArray);

    const averageLevel = dataArray.reduce((sum, value) => sum + value, 0) / bufferLength;
    const maxLevel = Math.max(...dataArray);
    const signalPresent = averageLevel > 5;
    const noiseLevel = dataArray.slice(0, 50).reduce((sum, value) => sum + value, 0) / 50;
    const signalToNoise = signalPresent ? averageLevel / (noiseLevel + 1) : 0;

    return {
      averageLevel: Math.round(averageLevel),
      maxLevel: Math.round(maxLevel),
      signalPresent,
      signalToNoise: Math.round(signalToNoise * 10) / 10,
      quality: signalToNoise > 3 ? 'good' : signalToNoise > 1.5 ? 'fair' : 'poor'
    };
  }, []);

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      setRecordingStatus('initializing');

      const constraints = {
        audio: {
          sampleRate: 44100,
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          latency: 0.01
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      await setupAudioContext(stream);

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 128000
      });

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { 
          type: 'audio/webm;codecs=opus' 
        });
        setAudioBlob(audioBlob);
        setHasRecording(true);
        setRecordingStatus('completed');
        
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
        
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
          audioContextRef.current.close();
          audioContextRef.current = null;
        }
        
        if (mediaRecorderRef.current?.qualityInterval) {
          clearInterval(mediaRecorderRef.current.qualityInterval);
        }
      };

      mediaRecorder.onerror = (event) => {
        console.error('MediaRecorder error:', event.error);
        setError(`Помилка запису: ${event.error?.message || 'Невідома помилка'}`);
        setRecordingStatus('error');
      };

      mediaRecorder.start(100);
      setIsRecording(true);
      setRecordingStatus('recording');
      startTimer();

      const qualityInterval = setInterval(() => {
        const quality = analyzeAudioQuality();
        setAudioQuality(quality);
      }, 500);

      mediaRecorderRef.current.qualityInterval = qualityInterval;

    } catch (error) {
      console.error('Error accessing microphone:', error);
      let errorMessage = 'Помилка доступу до мікрофона';
      
      if (error.name === 'NotAllowedError') {
        errorMessage = 'Доступ до мікрофона заборонено. Дозвольте використання мікрофона в налаштуваннях браузера.';
      } else if (error.name === 'NotFoundError') {
        errorMessage = 'Мікрофон не знайдено. Переконайтеся, що мікрофон підключено.';
      } else if (error.name === 'NotReadableError') {
        errorMessage = 'Мікрофон зайнятий іншою програмою або пошкоджений.';
      } else if (error.message) {
        errorMessage += `: ${error.message}`;
      }
      
      setError(errorMessage);
      setRecordingStatus('error');
    }
  }, [startTimer, setupAudioContext, analyzeAudioQuality]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      try {
        if (mediaRecorderRef.current.qualityInterval) {
          clearInterval(mediaRecorderRef.current.qualityInterval);
          mediaRecorderRef.current.qualityInterval = null;
        }

        if (mediaRecorderRef.current.state === 'recording') {
          mediaRecorderRef.current.stop();
        }
        
        setIsRecording(false);
        stopTimer();
        setAudioQuality(null);
        
      } catch (error) {
        console.error('Error stopping recording:', error);
        setError('Помилка зупинки запису');
      }
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
      try {
        const audioUrl = URL.createObjectURL(audioBlob);
        audioRef.current.src = audioUrl;
        
        audioRef.current.onloadstart = () => {
          console.log('Audio loading started');
        };
        
        audioRef.current.oncanplay = () => {
          console.log('Audio can play');
        };
        
        audioRef.current.onplay = () => {
          setIsPlaying(true);
        };

        audioRef.current.onended = () => {
          setIsPlaying(false);
          URL.revokeObjectURL(audioUrl);
        };
        
        audioRef.current.onerror = (e) => {
          console.error('Audio playback error:', e);
          setIsPlaying(false);
          URL.revokeObjectURL(audioUrl);
          setError('Помилка відтворення аудіо');
        };
        
        audioRef.current.onpause = () => {
          setIsPlaying(false);
        };

        const playPromise = audioRef.current.play();
        
        if (playPromise !== undefined) {
          playPromise
            .then(() => {
              console.log('Audio playback started successfully');
            })
            .catch(error => {
              console.error('Error playing audio:', error);
              setIsPlaying(false);
              URL.revokeObjectURL(audioUrl);
              setError('Не вдалося відтворити аудіо. Спробуйте ще раз.');
            });
        }
        
      } catch (error) {
        console.error('Error creating audio URL:', error);
        setError('Помилка підготовки аудіо для відтворення');
      }
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
    if (!audioBlob) {
      setError('Немає запису для аналізу');
      return;
    }
    
    setError(null);
    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      const audioBase64 = await audioToBase64(audioBlob);
      
      const requestData = {
        audio_data: audioBase64,
        format: 'webm',
        preprocessing_level: processingSettings.preprocessing_level,
        max_segments: processingSettings.max_segments
      };

      const response = await api.post('/api/recognition/interval/', requestData);
      
      if (response.data.status === 'completed') {
        setAnalysisResult(response.data.result);
      } else {
        throw new Error(response.data.message || 'Помилка аналізу');
      }

    } catch (error) {
      console.error('Error analyzing audio:', error);
      const errorMessage = error.response?.data?.error || 
                          error.response?.data?.message || 
                          'Помилка аналізу аудіо';
      setError(errorMessage);
    } finally {
      setIsAnalyzing(false);
    }
  }, [audioBlob, audioToBase64, processingSettings]);

  const quickAnalyze = useCallback(async () => {
    if (!audioBlob) return;
    
    setError(null);
    setIsAnalyzing(true);

    try {
      const audioBase64 = await audioToBase64(audioBlob);
      
      const requestData = {
        audio_data: audioBase64,
        format: 'webm'
      };

      const response = await api.post('/api/recognition/quick/', requestData);
      setAnalysisResult({
        status: 'success',
        mode: 'quick',
        best_prediction: response.data.prediction,
        alternative_predictions: response.data.alternatives || [],
        processing_info: {
          processing_time: response.data.processing_time,
          mode: 'Швидкий режим'
        }
      });

    } catch (error) {
      console.error('Error in quick analysis:', error);
      setError(error.response?.data?.error || 'Помилка швидкого аналізу');
    } finally {
      setIsAnalyzing(false);
    }
  }, [audioBlob, audioToBase64]);

  useEffect(() => {
    return () => {
      stopTimer();
      
      if (mediaRecorderRef.current) {
        if (mediaRecorderRef.current.qualityInterval) {
          clearInterval(mediaRecorderRef.current.qualityInterval);
        }
        if (mediaRecorderRef.current.state === 'recording') {
          mediaRecorderRef.current.stop();
        }
      }
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
      
      if (audioRef.current && audioRef.current.src) {
        URL.revokeObjectURL(audioRef.current.src);
      }
    };
  }, [stopTimer]);

  const getRecordingButtonClass = () => {
    const baseClass = "record-btn";
    if (isRecording) return `${baseClass} ${baseClass}--recording`;
    if (hasRecording) return `${baseClass} ${baseClass}--completed`;
    return baseClass;
  };

  const getQualityColor = (quality) => {
    switch (quality?.quality) {
      case 'good': return '#10b981';
      case 'fair': return '#f59e0b';
      case 'poor': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const renderResults = () => {
    return (
      <RecognitionResults 
        result={analysisResult}
        isAnalyzing={isAnalyzing}
        error={error}
      />
    );
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
                  {audioQuality && isRecording && (
                    <div className="quality-indicator">
                      <div 
                        className="quality-bar"
                        style={{ 
                          width: `${Math.min(audioQuality.averageLevel / 2, 100)}%`,
                          backgroundColor: getQualityColor(audioQuality)
                        }}
                      />
                      <span className="quality-text">{audioQuality.quality}</span>
                    </div>
                  )}
                </div>
              </div>

              <div className="audio-recognizer__controls">
                <button
                  className={getRecordingButtonClass()}
                  onClick={toggleRecording}
                  disabled={isAnalyzing}
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
                      disabled={isPlaying || isAnalyzing}
                      aria-label="Відтворити запис"
                    >
                      <Play />
                      <span>{isPlaying ? "Відтворюється..." : "Прослухати"}</span>
                    </button>

                    <button
                      className="action-btn action-btn--primary"
                      onClick={analyzeAudio}
                      disabled={isAnalyzing}
                      aria-label="Детальний аналіз"
                    >
                      {isAnalyzing ? <Loader className="spin" /> : <TrendingUp />}
                      <span>{isAnalyzing ? "Аналізуємо..." : "Детальний аналіз"}</span>
                    </button>

                    <button
                      className="action-btn action-btn--tertiary"
                      onClick={quickAnalyze}
                      disabled={isAnalyzing}
                      aria-label="Швидкий аналіз"
                    >
                      <Zap />
                      <span>Швидко</span>
                    </button>

                    <button
                      className="action-btn action-btn--ghost"
                      onClick={resetRecording}
                      disabled={isAnalyzing}
                      aria-label="Скинути запис"
                    >
                      <RotateCcw />
                      <span>Скинути</span>
                    </button>
                  </div>
                )}
              </div>

              <div className="processing-settings">
                <h3 className="processing-settings__title">
                  <Settings />
                  Налаштування обробки
                </h3>
                <div className="processing-settings__grid">
                  <div className="setting-group">
                    <label htmlFor="preprocessing">Рівень попередньої обробки:</label>
                    <select
                      id="preprocessing"
                      value={processingSettings.preprocessing_level}
                      onChange={(e) => setProcessingSettings(prev => ({
                        ...prev,
                        preprocessing_level: e.target.value
                      }))}
                      disabled={isRecording || isAnalyzing}
                    >
                      <option value="minimal">Мінімальний</option>
                      <option value="standard">Стандартний</option>
                      <option value="aggressive">Агресивний</option>
                    </select>
                  </div>
                  <div className="setting-group">
                    <label htmlFor="maxSegments">Макс. сегментів:</label>
                    <select
                      id="maxSegments"
                      value={processingSettings.max_segments}
                      onChange={(e) => setProcessingSettings(prev => ({
                        ...prev,
                        max_segments: parseInt(e.target.value)
                      }))}
                      disabled={isRecording || isAnalyzing}
                    >
                      <option value={1}>1</option>
                      <option value={2}>2</option>
                      <option value={3}>3</option>
                      <option value={5}>5</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>

            {renderResults()}

            {recordingStatus === 'completed' && !analysisResult && (
              <div className="audio-recognizer__status">
                <div className="status-card status-card--success">
                  <div className="status-card__icon">
                    <CheckCircle />
                  </div>
                  <div className="status-card__content">
                    <h3 className="status-card__title">Запис завершено</h3>
                    <p className="status-card__description">
                      Тривалість: {formatTime(seconds)}. Виберіть тип аналізу для розпізнавання інтервалу.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {recordingStatus === 'initializing' && (
              <div className="audio-recognizer__status">
                <div className="status-card status-card--info">
                  <div className="status-card__icon">
                    <Loader className="spin" />
                  </div>
                  <div className="status-card__content">
                    <h3 className="status-card__title">Ініціалізація</h3>
                    <p className="status-card__description">
                      Налаштування мікрофона та аудіо системи...
                    </p>
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className="audio-recognizer__status">
                <div className="status-card status-card--error">
                  <div className="status-card__icon">
                    <AlertCircle />
                  </div>
                  <div className="status-card__content">
                    <h3 className="status-card__title">Помилка</h3>
                    <p className="status-card__description">{error}</p>
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