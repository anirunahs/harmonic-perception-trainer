import React, { useEffect, useRef, useState, useCallback } from "react";
import { Mic, MicOff, Play, Square, RotateCcw } from "lucide-react";

const MicrophoneRecorder = ({ 
  onRecordingComplete, 
  onRecordingStart,
  onRecordingStop,
  maxDuration = 10,
  autoAnalyze = false,
  showPlayback = true,
  className = ""
}) => {
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
      setSeconds((prev) => {
        const newSeconds = prev + 1;
        if (newSeconds >= maxDuration) {
          stopRecording();
          return prev;
        }
        return newSeconds;
      });
    }, 1000);
  }, [maxDuration]);

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
    
    if (audioRef.current) {
      audioRef.current.src = '';
    }
  }, []);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        } 
      });
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setAudioBlob(audioBlob);
        setHasRecording(true);
        setRecordingStatus('completed');
        stream.getTracks().forEach(track => track.stop());
        
        if (onRecordingComplete) {
          const reader = new FileReader();
          reader.onloadend = () => {
            const base64Audio = reader.result.split(',')[1];
            onRecordingComplete(audioBlob, base64Audio);
          };
          reader.readAsDataURL(audioBlob);
        }
      };

      mediaRecorder.start(100);
      setIsRecording(true);
      setRecordingStatus('recording');
      startTimer();
      
      if (onRecordingStart) {
        onRecordingStart();
      }
      
    } catch (error) {
      console.error('Error accessing microphone:', error);
      setRecordingStatus('error');
      
      let errorMessage = 'Помилка доступу до мікрофона.';
      if (error.name === 'NotAllowedError') {
        errorMessage = 'Доступ до мікрофона заборонено. Дозвольте доступ у налаштуваннях браузера.';
      } else if (error.name === 'NotFoundError') {
        errorMessage = 'Мікрофон не знайдено. Перевірте підключення.';
      }
      
      alert(errorMessage);
    }
  }, [startTimer, onRecordingStart, onRecordingComplete, maxDuration]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      stopTimer();
      
      if (onRecordingStop) {
        onRecordingStop();
      }
    }
  }, [isRecording, stopTimer, onRecordingStop]);

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

  useEffect(() => {
    return () => {
      stopTimer();
      if (mediaRecorderRef.current && isRecording) {
        mediaRecorderRef.current.stop();
      }
    };
  }, [stopTimer, isRecording]);

  const getRecordingButtonClass = () => {
    const baseClass = "microphone-btn";
    if (isRecording) return `${baseClass} ${baseClass}--recording`;
    if (hasRecording) return `${baseClass} ${baseClass}--completed`;
    if (recordingStatus === 'error') return `${baseClass} ${baseClass}--error`;
    return baseClass;
  };

  const getStatusMessage = () => {
    switch (recordingStatus) {
      case 'recording':
        return `Запис... ${formatTime(seconds)} / ${formatTime(maxDuration)}`;
      case 'completed':
        return `Запис завершено (${formatTime(seconds)})`;
      case 'error':
        return 'Помилка запису';
      default:
        return 'Натисніть для початку запису';
    }
  };

  return (
    <div className={`microphone-recorder ${className}`}>
      <div className="microphone-recorder__display">
        <div className="microphone-recorder__timer">
          <span className="timer-text">{formatTime(seconds)}</span>
          {maxDuration > 0 && (
            <span className="timer-max">/ {formatTime(maxDuration)}</span>
          )}
          {isRecording && <div className="recording-indicator" />}
        </div>
        
        <div className="microphone-recorder__status">
          {getStatusMessage()}
        </div>
      </div>

      <div className="microphone-recorder__controls">
        <button
          className={getRecordingButtonClass()}
          onClick={toggleRecording}
          disabled={recordingStatus === 'error'}
          aria-label={isRecording ? "Зупинити запис" : "Почати запис"}
        >
          <div className="microphone-btn__icon">
            {isRecording ? <Square /> : <Mic />}
          </div>
          <span className="microphone-btn__text">
            {isRecording ? "Зупинити" : hasRecording ? "Новий запис" : "Записати"}
          </span>
        </button>

        {hasRecording && showPlayback && (
          <div className="microphone-recorder__playback">
            <button
              className="playback-btn"
              onClick={playRecording}
              disabled={isPlaying}
              aria-label="Прослухати запис"
            >
              <Play />
              <span>{isPlaying ? "Відтворюється..." : "Прослухати"}</span>
            </button>

            <button
              className="reset-btn"
              onClick={resetRecording}
              aria-label="Видалити запис"
            >
              <RotateCcw />
              <span>Видалити</span>
            </button>
          </div>
        )}
      </div>

      <audio ref={audioRef} style={{ display: 'none' }} />
    </div>
  );
};

export default MicrophoneRecorder;