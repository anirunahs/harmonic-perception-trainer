import React, { useEffect, useRef, useState, useCallback } from "react";
import { Mic, MicOff, Play, Square, RotateCcw, Volume2 } from "lucide-react";
import Header from "../components/Header";

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
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
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
      alert('Помилка доступу до мікрофона. Перевірте дозволи.');
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

  const analyzeAudio = useCallback(() => {
    if (!audioBlob) return;
    
    // TODO: Аудіоаналіз
    alert('Функція аналізу аудіо!');
  }, [audioBlob]);

  useEffect(() => {
    return () => {
      stopTimer();
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

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
            Запишіть звук та отримайте аналіз музичного інтервалу
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
        </div>

        <audio ref={audioRef} style={{ display: 'none' }} />
      </div>
    </div>
















      
    </>
  );
};

export default AudioRecognizer;
