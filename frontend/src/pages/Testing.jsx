import React, { useState, useEffect } from "react";
import {  ChevronLeft,  ChevronRight,  RotateCcw,  CheckCircle,  Trophy,  Star,} from "lucide-react";
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
  });
  const [showResults, setShowResults] = useState(false);

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

  const testTypes = [
    {
      id: "interval_recognition",
      name: "Розпізнавання інтервалів",
      description: "Прослухайте музичний інтервал та оберіть правильну назву",
      icon: "M",
      difficulty: "medium",
    },
    {
      id: "note_reproduction",
      name: "Відтворення нот",
      description: "Прослухайте ноту та відтворіть її голосом",
      icon: "Г",
      difficulty: "hard",
    },
  ];

  const intervals = [
    { id: "minor_second", name: "Мала секунда" },
    { id: "major_second", name: "Велика секунда" },
    { id: "minor_third", name: "Мала терція" },
    { id: "major_third", name: "Велика терція" },
    { id: "perfect_fourth", name: "Чиста кварта" },
    { id: "tritone", name: "Тритон" },
    { id: "perfect_fifth", name: "Чиста квінта" },
    { id: "minor_sixth", name: "Мала секста" },
    { id: "major_sixth", name: "Велика секста" },
    { id: "minor_seventh", name: "Мала септима" },
    { id: "major_seventh", name: "Велика септима" },
    { id: "perfect_octave", name: "Чиста октава" },
  ];

  const handleStartTest = async () => {
    if (!selectedTestType) return;

    try {
      await createTestSession(selectedTestType, testSettings);
    } catch (error) {
      console.error('Помилка створення тесту:', error);
    }
  };

  const handleReturnToMenu = () => {
    resetTest();
    setSelectedTestType(null);
    setShowResults(false);
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

  return (
    <>
      <Header />
      <div className="testing-page">
        <div className="testing-page__container">
          <div className="testing-page__header">
            <h1 className="testing-page__title">Тестування знань</h1>
            <p className="testing-page__subtitle">
              Оберіть тип тесту та налаштуйте параметри для перевірки ваших
              навичок
            </p>
          </div>

          <div className="testing-page__content">
            <div className="test-type-selection">
              <h2 className="section-title">Тип тесту</h2>

              <div className="test-types-grid">
                {testTypes.map((testType) => (
                  <button
                    key={testType.id}
                    className={`test-type-card ${
                      selectedTestType === testType.id
                        ? "test-type-card--selected"
                        : ""
                    }`}
                    onClick={() => setSelectedTestType(testType.id)}
                  >
                    <div className="test-type-card__icon">{testType.icon}</div>
                    <h3 className="test-type-card__name">{testType.name}</h3>
                    <p className="test-type-card__description">
                      {testType.description}
                    </p>
                    <div className="test-type-card__difficulty">
                      <Star />
                      <span>
                        Складність:{" "}
                        {testType.difficulty === "easy"
                          ? "Легка"
                          : testType.difficulty === "medium"
                          ? "Середня"
                          : "Важка"}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {selectedTestType && (
              <div className="test-settings">
                <h2 className="section-title">Налаштування</h2>

                <div className="settings-grid">
                  <div className="setting-group">
                    <label className="setting-label">Кількість питань</label>
                    <select
                      className="setting-select"
                      value={testSettings.totalQuestions}
                      onChange={(e) =>
                        setTestSettings((prev) => ({
                          ...prev,
                          totalQuestions: parseInt(e.target.value),
                        }))
                      }
                    >
                      <option value={5}>5 питань</option>
                      <option value={10}>10 питань</option>
                      <option value={15}>15 питань</option>
                      <option value={20}>20 питань</option>
                    </select>
                  </div>

                  <div className="setting-group">
                    <label className="setting-label">Складність</label>
                    <select
                      className="setting-select"
                      value={testSettings.difficulty}
                      onChange={(e) =>
                        setTestSettings((prev) => ({
                          ...prev,
                          difficulty: e.target.value,
                        }))
                      }
                    >
                      <option value="easy">Легка</option>
                      <option value="medium">Середня</option>
                      <option value="hard">Важка</option>
                    </select>
                  </div>

                  {/* Налаштування для тестів розпізнавання інтервалів */}
                  {selectedTestType === "interval_recognition" && (
                    <div className="setting-group setting-group--full">
                      <div className="setting-header">
                        <label className="setting-label">
                          Інтервали для тестування
                        </label>
                        <div className="setting-actions">
                          <button
                            className="btn btn--ghost btn--sm"
                            onClick={handleSelectAllIntervals}
                          >
                            Обрати всі
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
                        {intervals.map((interval) => (
                          <label
                            key={interval.id}
                            className="interval-checkbox"
                          >
                            <input
                              type="checkbox"
                              checked={testSettings.intervals.includes(
                                interval.id
                              )}
                              onChange={() => handleIntervalToggle(interval.id)}
                            />
                            <span className="interval-checkbox__checkmark"></span>
                            <span className="interval-checkbox__label">
                              {interval.name}
                            </span>
                          </label>
                        ))}
                      </div>

                      {testSettings.intervals.length === 0 && (
                        <div className="setting-warning">
                          Оберіть хоча б один інтервал для тестування
                        </div>
                      )}
                    </div>
                  )}

                  {/* Інформація про тест відтворення нот */}
                  {selectedTestType === "note_reproduction" && (
                    <div className="setting-group setting-group--full">
                      <div className="setting-info">
                        <h4>Інформація про тест</h4>
                        <p>
                          Тест використовуватиме ваш налаштований вокальний
                          діапазон. Якщо ви ще не налаштували його, будуть
                          використані стандартні ноти (C3-C5).
                        </p>
                        <p>
                          <strong>Рекомендація:</strong> Налаштуйте свій
                          вокальний діапазон у профілі для більш комфортного
                          тестування.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {selectedTestType && (
              <div className="test-start">
                <button
                  className="btn btn--primary btn--lg"
                  onClick={handleStartTest}
                  disabled={
                    isLoading ||
                    (selectedTestType === "interval_recognition" &&
                      testSettings.intervals.length === 0)
                  }
                >
                  {isLoading ? (
                    <>
                      <LoadingIndicator size="small" />
                      <span>Створення тесту...</span>
                    </>
                  ) : (
                    <>
                      <CheckCircle />
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
