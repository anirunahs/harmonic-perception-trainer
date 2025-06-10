import React, { useState, useEffect } from "react";
import { ChevronLeft, ChevronRight, RotateCcw, CheckCircle, Trophy, Star, Settings, Info, Timer, Target, 
         Award, BookOpen, Play, Pause} from "lucide-react";
import Header from "../components/Header";
import IntervalRecognitionTest from "../components/testing/IntervalRecognitionTest";
import { useTesting } from "../hooks/useTesting";
import LoadingIndicator from "../components/LoadingIndicator";

const Testing = () => {
  const [selectedTestType, setSelectedTestType] = useState(null);
  const [testSettings, setTestSettings] = useState({
    totalQuestions: 10,
    intervals: [],
    difficulty: "medium",
    timeLimit: null,
    showProgress: true,
    playbackSpeed: 1.0,
    autoNext: false
  });
  const [showResults, setShowResults] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [timeLeft, setTimeLeft] = useState(null);
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
  } = useTesting();

  useEffect(() => {
    if (currentSession && testSettings.timeLimit && !isPaused && !sessionCompleted) {
      const timer = setInterval(() => {
        setTimeLeft(prev => {
          if (prev <= 1) {
            handleTimeUp();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);

      return () => clearInterval(timer);
    }
  }, [currentSession, testSettings.timeLimit, isPaused, sessionCompleted]);

  const testTypes = [
    {
      id: "interval_recognition",
      name: "Розпізнавання інтервалів",
      description: "Прослухайте музичний інтервал та оберіть правильну назву",
      icon: "🎵",
      difficulty: "medium",
      estimatedTime: "5-15 хв",
      skills: ["Слух", "Теорія музики", "Розпізнавання"],
      tips: "Почніть з простих інтервалів як кварта та квінта"
    },
    {
      id: "note_reproduction",
      name: "Відтворення нот",
      description: "Прослухайте ноту та відтворіть її голосом",
      icon: "🎤",
      difficulty: "hard",
      estimatedTime: "10-20 хв",
      skills: ["Вокал", "Інтонація", "Музичний слух"],
      tips: "Налаштуйте свій вокальний діапазон для кращих результатів"
    },
  ];

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

  useEffect(() => {
    if (currentSession && testSettings.timeLimit) {
      setTimeLeft(testSettings.timeLimit);
    }
  }, [currentSession, testSettings.timeLimit]);

  const handleTimeUp = () => {
    console.log("Час вийшов!");
  };

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

  const handleStartTest = async () => {
    if (!selectedTestType) return;

    try {
      await createTestSession(selectedTestType, testSettings);
      if (testSettings.timeLimit) {
        setTimeLeft(testSettings.timeLimit);
      }
    } catch (error) {
      console.error('Помилка створення тесту:', error);
    }
  };

  const handleAnswerSubmit = async (answer, recordedFrequency = null) => {
    try {
      const result = await submitAnswer(answer, recordedFrequency);
      
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
    setTimeLeft(null);
    setIsPaused(false);
  };

  const handleIntervalToggle = (intervalId) => {
    setTestSettings(prev => ({
      ...prev,
      intervals: prev.intervals.includes(intervalId)
        ? prev.intervals.filter(id => id !== intervalId)
        : [...prev.intervals, intervalId]
    }));
  };

  const handleSelectAllIntervals = () => {
    setTestSettings(prev => ({
      ...prev,
      intervals: intervals.map(interval => interval.id)
    }));
  };

  const handleClearIntervals = () => {
    setTestSettings(prev => ({
      ...prev,
      intervals: []
    }));
  };

  const handleDifficultyPreset = (difficulty) => {
    setTestSettings(prev => ({
      ...prev,
      intervals: difficultyPresets[difficulty] || [],
      difficulty
    }));
  };

  const togglePause = () => {
    setIsPaused(prev => !prev);
  };

  const progress = getTestProgress();
  const correctAnswers = getCorrectAnswersCount();

  // Результат
  if (showResults && results) {
    const accuracy = results.accuracy_percentage;
    const getPerformanceLevel = () => {
      if (accuracy >= 90) return { level: "Відмінно", color: "text-success", icon: "🏆" };
      if (accuracy >= 80) return { level: "Добре", color: "text-primary", icon: "⭐" };
      if (accuracy >= 70) return { level: "Задовільно", color: "text-warning", icon: "👍" };
      return { level: "Потребує покращення", color: "text-error", icon: "📚" };
    };

    const performance = getPerformanceLevel();

    return (
      <>
        <Header />
        <div className="testing-page">
          <div className="testing-page__container">
            <div className="test-results">
              <div className="test-results__header">
                <div className="results-performance">
                  <span className="performance-icon">{performance.icon}</span>
                  <h1 className="results-title">{performance.level}</h1>
                </div>
                <p className="results-subtitle">
                  {testTypes.find(t => t.id === results.test_type)?.name}
                </p>
                <div className="results-score">
                  <span className="score-value">{results.accuracy_percentage.toFixed(1)}%</span>
                  <span className="score-label">Точність</span>
                </div>
              </div>

              <div className="test-results__stats">
                <div className="stats-grid">
                  <div className="stat-card">
                    <CheckCircle className="stat-icon text-success" />
                    <div className="stat-value">{results.correct_answers}</div>
                    <div className="stat-label">Правильних відповідей</div>
                  </div>
                  
                  <div className="stat-card">
                    <Target className="stat-icon text-primary" />
                    <div className="stat-value">{results.total_questions}</div>
                    <div className="stat-label">Загальна кількість</div>
                  </div>
                  
                  <div className="stat-card">
                    <Award className="stat-icon text-warning" />
                    <div className="stat-value">+{results.experience_gained}</div>
                    <div className="stat-label">Досвід</div>
                  </div>
                  
                  {testSettings.timeLimit && (
                    <div className="stat-card">
                      <Timer className="stat-icon text-secondary" />
                      <div className="stat-value">{formatTime(testSettings.timeLimit - (timeLeft || 0))}</div>
                      <div className="stat-label">Витрачено часу</div>
                    </div>
                  )}
                </div>

                <div className="accuracy-bar">
                  <div 
                    className="accuracy-bar__fill" 
                    style={{ 
                      width: `${results.accuracy_percentage}%`,
                      background: accuracy >= 80 ? 'linear-gradient(90deg, #10b981, #059669)' : 
                                 accuracy >= 70 ? 'linear-gradient(90deg, #f59e0b, #d97706)' :
                                 'linear-gradient(90deg, #ef4444, #dc2626)'
                    }} 
                  />
                  <span className="accuracy-bar__text">{results.accuracy_percentage.toFixed(1)}%</span>
                </div>

                {selectedTestType === 'interval_recognition' && (
                  <div className="detailed-stats">
                    <h3>Результати по інтервалах</h3>
                    <div className="interval-stats">
                      {Object.entries(
                        results.questions?.reduce((acc, q) => {
                          const intervalName = intervals.find(i => i.id === q.interval_type)?.name || q.interval_type;
                          if (!acc[intervalName]) acc[intervalName] = { correct: 0, total: 0 };
                          acc[intervalName].total++;
                          if (answers[q.id]?.is_correct) acc[intervalName].correct++;
                          return acc;
                        }, {}) || {}
                      ).map(([interval, stats]) => (
                        <div key={interval} className="interval-stat">
                          <span className="interval-name">{interval}</span>
                          <span className="interval-result">
                            {stats.correct}/{stats.total} ({Math.round((stats.correct/stats.total)*100)}%)
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="test-results__actions">
                <button
                  className="btn btn--primary btn--lg"
                  onClick={() => {
                    setShowResults(false);
                    handleStartTest();
                  }}
                >
                  <RotateCcw />
                  <span>Пройти ще раз</span>
                </button>

                <button
                  className="btn btn--ghost btn--lg"
                  onClick={handleReturnToMenu}
                >
                  <ChevronLeft />
                  <span>Назад до меню</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </>
    );
  }

  if (currentSession && currentQuestion) {
    return (
      <>
        <Header />
        <div className="testing-page">
          <div className="testing-page__container">
            <div className="test-session">
              <div className="test-session__header">
                <div className="test-info">
                  <h1 className="test-title">
                    {testTypes.find(t => t.id === currentSession.test_type)?.name}
                  </h1>
                  <div className="test-meta">
                    <div className="test-progress">
                      <span className="progress-text">
                        Питання {questionIndex + 1} з {currentSession.total_questions}
                      </span>
                      <div className="progress-bar">
                        <div 
                          className="progress-bar__fill" 
                          style={{ width: `${((questionIndex + 1) / currentSession.total_questions) * 100}%` }} 
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
                          onClick={togglePause}
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
                  onClick={handleReturnToMenu}
                >
                  <ChevronLeft />
                  <span>Вийти з тесту</span>
                </button>
              </div>

              <div className="test-session__content">
                {currentSession.test_type === 'interval_recognition' ? (
                  <IntervalRecognitionTest
                    question={currentQuestion}
                    onAnswer={handleAnswerSubmit}
                    isSubmitting={isLoading}
                    showFeedback={true}
                    autoNext={testSettings.autoNext}
                  />
                ) : currentSession.test_type === 'note_reproduction' ? (
                  <div className="note-reproduction-placeholder">
                    <h3>Тест відтворення нот</h3>
                    <p>Цей тип тесту ще не реалізований</p>
                  </div>
                ) : null}
              </div>

              <div className="test-session__navigation">
                <button
                  className="btn btn--ghost"
                  onClick={previousQuestion}
                  disabled={questionIndex === 0}
                >
                  <ChevronLeft />
                  <span>Попереднє</span>
                </button>

                <div className="question-indicator">
                  <span className="current-question">{questionIndex + 1}</span>
                  <span className="separator">/</span>
                  <span className="total-questions">{currentSession.total_questions}</span>
                </div>

                <button
                  className="btn btn--ghost"
                  onClick={nextQuestion}
                  disabled={questionIndex >= currentSession.total_questions - 1}
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
            <div className="test-type-selection">
              <h2 className="section-title">
                <BookOpen />
                Оберіть тип тесту
              </h2>
              
              <div className="test-types-grid">
                {testTypes.map((testType) => (
                  <button
                    key={testType.id}
                    className={`test-type-card ${
                      selectedTestType === testType.id ? 'test-type-card--selected' : ''
                    }`}
                    onClick={() => setSelectedTestType(testType.id)}
                  >
                    <div className="test-type-card__icon">{testType.icon}</div>
                    <h3 className="test-type-card__name">{testType.name}</h3>
                    <p className="test-type-card__description">{testType.description}</p>
                    
                    <div className="test-type-card__meta">
                      <div className="test-meta-item">
                        <Timer className="meta-icon" />
                        <span>{testType.estimatedTime}</span>
                      </div>
                      <div className="test-meta-item">
                        <Star className="meta-icon" />
                        <span>{testType.difficulty === 'easy' ? 'Легкий' : testType.difficulty === 'medium' ? 'Середній' : 'Складний'}</span>
                      </div>
                    </div>

                    <div className="test-type-card__skills">
                      <span className="skills-label">Навички:</span>
                      <div className="skills-list">
                        {testType.skills.map(skill => (
                          <span key={skill} className="skill-tag">{skill}</span>
                        ))}
                      </div>
                    </div>

                    <div className="test-type-card__tip">
                      <Info className="tip-icon" />
                      <span>{testType.tips}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {selectedTestType && (
              <div className="test-settings">
                <div className="settings-header">
                  <h2 className="section-title">
                    <Settings />
                    Налаштування тесту
                  </h2>
                  <button 
                    className="btn btn--ghost btn--sm"
                    onClick={() => setShowSettings(!showSettings)}
                  >
                    {showSettings ? 'Приховати' : 'Показати'} додаткові налаштування
                  </button>
                </div>
                
                <div className="settings-grid">
                  <div className="setting-group">
                    <label className="setting-label">Кількість питань</label>
                    <select
                      className="setting-select"
                      value={testSettings.totalQuestions}
                      onChange={(e) => setTestSettings(prev => ({
                        ...prev,
                        totalQuestions: parseInt(e.target.value)
                      }))}
                    >
                      <option value={5}>5 питань (швидкий тест)</option>
                      <option value={10}>10 питань (стандарт)</option>
                      <option value={15}>15 питань (детальний)</option>
                      <option value={20}>20 питань (повний)</option>
                    </select>
                  </div>

                  <div className="setting-group">
                    <label className="setting-label">Складність</label>
                    <select
                      className="setting-select"
                      value={testSettings.difficulty}
                      onChange={(e) => {
                        const difficulty = e.target.value;
                        setTestSettings(prev => ({
                          ...prev,
                          difficulty,
                          intervals: selectedTestType === 'interval_recognition' ? 
                            difficultyPresets[difficulty] || prev.intervals : prev.intervals
                        }));
                      }}
                    >
                      <option value="easy">Легка (основні інтервали)</option>
                      <option value="medium">Середня (розширений набір)</option>
                      <option value="hard">Важка (всі інтервали)</option>
                    </select>
                  </div>

                  {showSettings && (
                    <>
                      <div className="setting-group">
                        <label className="setting-label">Ліміт часу</label>
                        <select
                          className="setting-select"
                          value={testSettings.timeLimit || ''}
                          onChange={(e) => setTestSettings(prev => ({
                            ...prev,
                            timeLimit: e.target.value ? parseInt(e.target.value) : null
                          }))}
                        >
                          <option value="">Без ліміту</option>
                          <option value={300}>5 хвилин</option>
                          <option value={600}>10 хвилин</option>
                          <option value={900}>15 хвилин</option>
                          <option value={1200}>20 хвилин</option>
                        </select>
                      </div>

                      <div className="setting-group">
                        <label className="setting-checkbox">
                          <input
                            type="checkbox"
                            checked={testSettings.autoNext}
                            onChange={(e) => setTestSettings(prev => ({
                              ...prev,
                              autoNext: e.target.checked
                            }))}
                          />
                          <span className="checkmark"></span>
                          Автоматичний перехід до наступного питання
                        </label>
                      </div>

                      <div className="setting-group">
                        <label className="setting-checkbox">
                          <input
                            type="checkbox"
                            checked={testSettings.showProgress}
                            onChange={(e) => setTestSettings(prev => ({
                              ...prev,
                              showProgress: e.target.checked
                            }))}
                          />
                          <span className="checkmark"></span>
                          Показувати прогрес виконання
                        </label>
                      </div>
                    </>
                  )}

                  {selectedTestType === 'interval_recognition' && (
                    <div className="setting-group setting-group--full">
                      <div className="setting-header">
                        <label className="setting-label">Інтервали для тестування</label>
                        <div className="setting-actions">
                          <button 
                            className="btn btn--ghost btn--sm"
                            onClick={() => handleDifficultyPreset('easy')}
                          >
                            Легкі
                          </button>
                          <button 
                            className="btn btn--ghost btn--sm"
                            onClick={() => handleDifficultyPreset('medium')}
                          >
                            Середні
                          </button>
                          <button 
                            className="btn btn--ghost btn--sm"
                            onClick={() => handleDifficultyPreset('hard')}
                          >
                            Всі
                          </button>
                          <button 
                            className="btn btn--ghost btn--sm"
                            onClick={handleClearIntervals}
                          >
                            Очистити
                          </button>
                        </div>
                      </div>
                      
                      <div className="intervals-selection">
                        {intervals.map(interval => (
                          <label key={interval.id} className="interval-checkbox">
                            <input
                              type="checkbox"
                              checked={testSettings.intervals.includes(interval.id)}
                              onChange={() => handleIntervalToggle(interval.id)}
                            />
                            <span className="interval-checkbox__checkmark"></span>
                            <div className="interval-info">
                              <span className="interval-name">{interval.name}</span>
                              <span className="interval-meta">
                                {interval.semitones} пт •
                                <span className={`difficulty difficulty--${interval.difficulty}`}>
                                  {interval.difficulty === 'easy' ? 'Легкий' : 
                                   interval.difficulty === 'medium' ? 'Середній' : 'Складний'}
                                </span>
                              </span>
                            </div>
                          </label>
                        ))}
                      </div>
                      
                      {testSettings.intervals.length === 0 && (
                        <div className="setting-warning">
                          <Info className="warning-icon" />
                          Оберіть хоча б один інтервал для тестування
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}

            {selectedTestType && (
              <div className="test-start">
                <div className="test-summary">
                  <div className="summary-item">
                    <Target className="summary-icon" />
                    <span>{testSettings.totalQuestions} питань</span>
                  </div>
                  {testSettings.timeLimit && (
                    <div className="summary-item">
                      <Timer className="summary-icon" />
                      <span>{formatTime(testSettings.timeLimit)}</span>
                    </div>
                  )}
                  {selectedTestType === 'interval_recognition' && (
                    <div className="summary-item">
                      <Star className="summary-icon" />
                      <span>{testSettings.intervals.length} інтервалів</span>
                    </div>
                  )}
                </div>

                <button
                  className="btn btn--primary btn--lg start-test-btn"
                  onClick={handleStartTest}
                  disabled={
                    isLoading || 
                    (selectedTestType === 'interval_recognition' && testSettings.intervals.length === 0)
                  }
                >
                  {isLoading ? (
                    <>
                      <LoadingIndicator size="small" />
                      <span>Створення тесту...</span>
                    </>
                  ) : (
                    <>
                      <Play />
                      <span>Розпочати тест</span>
                    </>
                  )}
                </button>
              </div>
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