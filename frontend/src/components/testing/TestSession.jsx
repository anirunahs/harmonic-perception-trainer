import React from "react";
import { ChevronLeft, ChevronRight, Timer, Play, Pause } from "lucide-react";
import IntervalRecognitionTest from "./IntervalRecognitionTest";

const TestSession = ({
  session,
  currentQuestion,
  questionIndex,
  timeLeft,
  isPaused,
  testSettings,
  intervals,
  isLoading,
  error,
  onAnswerSubmit,
  onNextQuestion,
  onPreviousQuestion,
  onTogglePause,
  onReturnToMenu,
  setError
}) => {
  const formatTime = (seconds) => {
    if (!seconds) return "";
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getTimerColor = () => {
    if (!timeLeft || !testSettings.timeLimit) return "text-primary";
    const percentage = (timeLeft / testSettings.timeLimit) * 100;
    if (percentage <= 20) return "text-error";
    if (percentage <= 50) return "text-warning";
    return "text-primary";
  };

  const testTypes = {
    'interval_recognition': 'Розпізнавання інтервалів',
    'note_reproduction': 'Відтворення нот'
  };

  return (
    <div className="test-session">
      <div className="test-session__header">
        <div className="test-info">
          <h1 className="test-title">
            {testTypes[session.test_type] || session.test_type}
          </h1>
          <div className="test-meta">
            <div className="test-progress">
              <span className="progress-text">
                Питання {questionIndex + 1} з {session.total_questions}
              </span>
              <div className="progress-bar">
                <div 
                  className="progress-bar__fill" 
                  style={{ width: `${((questionIndex + 1) / session.total_questions) * 100}%` }} 
                />
              </div>
            </div>
            
            {testSettings.timeLimit && (
              <div className="test-timer">
                <Timer className={`timer-icon ${getTimerColor()}`} />
                <span className={`timer-text ${getTimerColor()}`}>
                  {formatTime(timeLeft)}
                </span>
                <button 
                  className="btn btn--ghost btn--sm"
                  onClick={onTogglePause}
                  title={isPaused ? "Продовжити" : "Пауза"}
                >
                  {isPaused ? <Play /> : <Pause />}
                </button>
              </div>
            )}
          </div>
        </div>

        <button
          className="btn btn--ghost"
          onClick={onReturnToMenu}
        >
          <ChevronLeft />
          <span>Вийти з тесту</span>
        </button>
      </div>

      <div className="test-session__content">
        {session.test_type === 'interval_recognition' ? (
          <IntervalRecognitionTest
            question={currentQuestion}
            onAnswer={onAnswerSubmit}
            isSubmitting={isLoading}
            showFeedback={true}
            autoNext={testSettings.autoNext}
            intervals={intervals}
            randomOrder={testSettings.randomOrder}
            includeReferenceNote={testSettings.includeReferenceNote}
            onError={setError}
          />
        ) : session.test_type === 'note_reproduction' ? (
          <div className="note-reproduction-placeholder">
            <h3>Тест відтворення нот</h3>
            <p>Цей тип тесту ще не реалізований</p>
          </div>
        ) : null}
      </div>

      <div className="test-session__navigation">
        <button
          className="btn btn--ghost"
          onClick={onPreviousQuestion}
          disabled={questionIndex === 0}
        >
          <ChevronLeft />
          <span>Попереднє</span>
        </button>

        <div className="question-indicator">
          <span className="current-question">{questionIndex + 1}</span>
          <span className="separator">/</span>
          <span className="total-questions">{session.total_questions}</span>
        </div>

        <button
          className="btn btn--ghost"
          onClick={onNextQuestion}
          disabled={questionIndex >= session.total_questions - 1}
        >
          <span>Наступне</span>
          <ChevronRight />
        </button>
      </div>

      {error && (
        <div className="test-session__error">
          <div className="alert alert--error">
            <span>{error}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default TestSession;