import React, { useEffect, useRef, useState, useCallback } from "react";
import { Mic, MicOff, Play, Square, RotateCcw, Volume2 } from "lucide-react";
import Header from "../components/Header";

const AudioRecognizer = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const timerRef = useRef(null);

  const formatTime = (s) => {
    const min = String(Math.floor(s / 60)).padStart(2, "0");
    const sec = String(s % 60).padStart(2, "0");
    return `${min}:${sec}`;
  };

  const startTimer = () => {
    if (timerRef.current) return;
    timerRef.current = setInterval(() => {
      setSeconds((prev) => prev + 1);
    }, 1000);
  };

  const stopTimer = () => {
    clearInterval(timerRef.current);
    timerRef.current = null;
  };

  const handleToggleRecording = () => {
    if (isRecording) {
      stopTimer();
    } else {
      setSeconds(0);
      startTimer();
    }
    setIsRecording((prev) => !prev);
  };

  useEffect(() => {
    return () => {
      stopTimer();
    };
  }, []);

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
              </div>
            </div>

            <div className="audio-recognizer__controls">
              <button className="record-btn">
                <div className="record-btn__icon">
                  <Mic />
                </div>
                <span className="record-btn__text">
                  Почати запис
                </span>
              </button>

              <div className="audio-recognizer__actions">
                <button className="action-btn action-btn--secondary" disabled>
                   <Play />
                   <span>Прослухати</span>
                 </button>
                 <button className="action-btn action-btn--primary" disabled>
                   <Volume2 />
                   <span>Аналізувати</span>
                 </button>
                 <button className="action-btn action-btn--ghost" disabled>
                   <RotateCcw />
                   <span>Скинути</span>
                 </button>
              </div>
            </div>
          </div>
        </div>
        <audio style={{ display: "none" }} />
      </div>
    </div>
















      
    </>
  );
};

export default AudioRecognizer;
