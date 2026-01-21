import React, { useState, useEffect } from "react";
import { ChevronLeft, RotateCcw, Play } from "lucide-react";
import Header from "../components/Header";
import TestTypeSelection from "../components/testing/TestTypeSelection";
import TestSettings from "../components/testing/TestSettings";
import TestSession from "../components/testing/TestSession";
import TestResults from "../components/testing/TestResults";
import { useTesting } from "../hooks/useTesting";
import LoadingIndicator from "../components/LoadingIndicator";

const Testing = () => {
  const [selectedTestType, setSelectedTestType] = useState(null);
  const [testSettings, setTestSettings] = useState({
    totalQuestions: 10,
    intervals: [],
    difficulty: "medium",
    instrument: "piano",
    timeLimit: null,
    enableHints: false,
    showProgress: true,
    autoNext: false,
    randomOrder: false,
    includeReferenceNote: true,
  });
  const [showResults, setShowResults] = useState(false);
  const [isPaused, setIsPaused] = useState(false);

  const {
    currentSession,
    currentQuestion,
    questionIndex,
    answers,
    isLoading,
    error,
    sessionCompleted,
    results,
    createTestSession,
    submitAnswer,
    nextQuestion,
    previousQuestion,
    resetTest,
    getTestProgress,
    getCorrectAnswersCount,
    setError,
    timeElapsed,
    questionTimeLeft,
  } = useTesting();

  const intervals = [
    { id: "minor_second", name: "Мала секунда", difficulty: "hard", semitones: 1 },
    { id: "major_second", name: "Велика секунда", difficulty: "medium", semitones: 2 },
    { id: "minor_third", name: "Мала терція", difficulty: "easy", semitones: 3 },
    { id: "major_third", name: "Велика терція", difficulty: "easy", semitones: 4 },
    { id: "perfect_fourth", name: "Чиста кварта", difficulty: "easy", semitones: 5 },
    { id: "tritone", name: "Тритон", difficulty: "hard", semitones: 6 },
    { id: "perfect_fifth", name: "Чиста квінта", difficulty: "easy", semitones: 7 },
    { id: "minor_sixth", name: "Мала секста", difficulty: "medium", semitones: 8 },
    { id: "major_sixth", name: "Велика секста", difficulty: "medium", semitones: 9 },
    { id: "minor_seventh", name: "Мала септима", difficulty: "hard", semitones: 10 },
    { id: "major_seventh", name: "Велика септима", difficulty: "hard", semitones: 11 },
    { id: "perfect_octave", name: "Чиста октава", difficulty: "easy", semitones: 12 },
  ];

  const difficultyPresets = {
    easy: intervals.filter(i => i.difficulty === "easy").map(i => i.id),
    medium: intervals.filter(i => ["easy", "medium"].includes(i.difficulty)).map(i => i.id),
    hard: intervals.map(i => i.id)
  };

  useEffect(() => {
    if (sessionCompleted && results) {
      setShowResults(true);
    }
  }, [sessionCompleted, results]);

  // Time left is now managed by useTestTimer hook

  const handleStartTest = async () => {
    if (!selectedTestType) return;

    try {
      await createTestSession(selectedTestType, testSettings);
    } catch (error) {
      console.error('Помилка створення тесту:', error);
    }
  };

  const handleAnswerSubmit = async (answer) => {
    try {
      const result = await submitAnswer(answer);
      
      if (testSettings.autoNext && !result.session_completed) {
        setTimeout(() => {
          nextQuestion();
        }, 1500);
      }
    } catch (error) {
      console.error('Помилка відправки відповіді:', error);
    }
  };

  const handleReturnToMenu = () => {
    resetTest();
    setSelectedTestType(null);
    setShowResults(false);
    setIsPaused(false);
  };

  const handleUpdateSettings = (newSettings) => {
    setTestSettings(prev => ({ ...prev, ...newSettings }));
  };

  const togglePause = () => {
    setIsPaused(prev => !prev);
  };

  // Результат
  if (showResults && results) {
    return (
      <>
        <Header />
        <div className="testing-page">
          <div className="testing-page__container">
            <TestResults
              results={results}
              testType={selectedTestType}
              intervals={intervals}
              answers={answers}
              onRetry={() => {
                setShowResults(false);
                handleStartTest();
              }}
              onReturnToMenu={handleReturnToMenu}
            />
          </div>
        </div>
      </>
    );
  }

  // Активний тест
  if (currentSession && currentQuestion) {
    return (
      <>
        <Header />
        <div className="testing-page">
          <div className="testing-page__container">
            <TestSession
              session={currentSession}
              currentQuestion={currentQuestion}
              questionIndex={questionIndex}
              timeLeft={questionTimeLeft}
              isPaused={isPaused}
              testSettings={testSettings}
              intervals={intervals}
              isLoading={isLoading}
              error={error}
              onAnswerSubmit={handleAnswerSubmit}
              onNextQuestion={nextQuestion}
              onPreviousQuestion={previousQuestion}
              onTogglePause={togglePause}
              onReturnToMenu={handleReturnToMenu}
              setError={setError}
            />
          </div>
        </div>
      </>
    );
  }

  // Меню вибору тесту та налаштувань
  return (
    <>
      <Header />
      <div className="testing-page">
        <div className="testing-page__container">
          <div className="testing-page__header">
            <h1 className="testing-page__title">Тестування знань</h1>
            <p className="testing-page__subtitle">
              Перевірте свої музичні навички та отримайте детальний аналіз результатів
            </p>
          </div>

          <div className="testing-page__content">
            <TestTypeSelection
              selectedTestType={selectedTestType}
              onSelectTestType={setSelectedTestType}
            />

            {selectedTestType && (
              <TestSettings
                testType={selectedTestType}
                settings={testSettings}
                intervals={intervals}
                difficultyPresets={difficultyPresets}
                onUpdateSettings={handleUpdateSettings}
                isLoading={isLoading}
                onStartTest={handleStartTest}
              />
            )}

            {error && (
              <div className="testing-page__error">
                <div className="alert alert--error">
                  <span>{error}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
};

export default Testing;