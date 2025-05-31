import React, { useEffect, useRef, useState } from "react";
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
      <div className="audio-recognizer-page">
        
        <h2>Розпізнавання інтервалу</h2>

        <div className="audio-timer">{formatTime(seconds)}</div>

        <button
          className={`btn--record-toggle ${isRecording ? "recording" : "idle"}`}
          onClick={handleToggleRecording}
        >
          {isRecording ? "Зупинити запис" : "Почати запис"}
        </button>

        <button className="btn--playback" disabled>
          Відтворити запис
        </button>
      </div>
    </>
  );
};

export default AudioRecognizer;
