import React, { useState } from "react";
import { Settings, Target, Timer, Star, Info, Play } from "lucide-react";
import LoadingIndicator from "../LoadingIndicator";

const TestSettings = ({ 
  testType, 
  settings, 
  intervals, 
  difficultyPresets, 
  onUpdateSettings, 
  isLoading, 
  onStartTest 
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleIntervalToggle = (intervalId) => {
    const newIntervals = settings.intervals.includes(intervalId)
      ? settings.intervals.filter(id => id !== intervalId)
      : [...settings.intervals, intervalId];
    
    onUpdateSettings({ intervals: newIntervals });
  };

  const handleSelectAllIntervals = () => {
    onUpdateSettings({ intervals: intervals.map(interval => interval.id) });
  };

  const handleClearIntervals = () => {
    onUpdateSettings({ intervals: [] });
  };

  const handleDifficultyPreset = (difficulty) => {
    const presetIntervals = difficultyPresets[difficulty] || [];
    onUpdateSettings({ 
      intervals: presetIntervals,
      difficulty 
    });
  };

  const formatTime = (seconds) => {
    if (!seconds) return "";
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const canStartTest = () => {
    if (testType === 'interval_recognition' && settings.intervals.length === 0) {
      return false;
    }
    return true;
  };

  return (
    <div className="test-settings">
      <div className="settings-header">
        <h2 className="section-title">
          <Settings />
          Налаштування тесту
        </h2>
        <button 
          className="btn btn--ghost btn--sm"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          {showAdvanced ? 'Приховати' : 'Показати'} додаткові налаштування
        </button>
      </div>
      
      <div className="settings-grid">
        {/* Основні налаштування */}
        <div className="setting-group">
          <label className="setting-label">Кількість питань</label>
          <select
            className="setting-select"
            value={settings.totalQuestions}
            onChange={(e) => onUpdateSettings({ totalQuestions: parseInt(e.target.value) })}
          >
            <option value={5}>5 питань (швидкий тест)</option>
            <option value={10}>10 питань (стандарт)</option>
            <option value={15}>15 питань (детальний)</option>
            <option value={20}>20 питань (повний)</option>
          </select>
        </div>

        <div className="setting-group">
          <label className="setting-label">Загальна складність</label>
          <select
            className="setting-select"
            value={settings.difficulty}
            onChange={(e) => {
              const difficulty = e.target.value;
              handleDifficultyPreset(difficulty);
            }}
          >
            <option value="easy">Легка (основні інтервали)</option>
            <option value="medium">Середня (розширений набір)</option>
            <option value="hard">Важка (всі інтервали)</option>
            <option value="custom">Власний вибір</option>
          </select>
        </div>

        {/* Додаткові налаштування */}
        {showAdvanced && (
          <>
            <div className="setting-group">
              <label className="setting-label">Ліміт часу</label>
              <select
                className="setting-select"
                value={settings.timeLimit || ''}
                onChange={(e) => onUpdateSettings({ 
                  timeLimit: e.target.value ? parseInt(e.target.value) : null 
                })}
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
                  checked={settings.autoNext}
                  onChange={(e) => onUpdateSettings({ autoNext: e.target.checked })}
                />
                <span className="checkmark"></span>
                Автоматичний перехід до наступного питання
              </label>
            </div>

            <div className="setting-group">
              <label className="setting-checkbox">
                <input
                  type="checkbox"
                  checked={settings.randomOrder}
                  onChange={(e) => onUpdateSettings({ randomOrder: e.target.checked })}
                />
                <span className="checkmark"></span>
                Випадковий порядок нот в інтервалі
              </label>
            </div>

            <div className="setting-group">
              <label className="setting-checkbox">
                <input
                  type="checkbox"
                  checked={settings.includeReferenceNote}
                  onChange={(e) => onUpdateSettings({ includeReferenceNote: e.target.checked })}
                />
                <span className="checkmark"></span>
                Включати референсну ноту перед інтервалом
              </label>
            </div>

            <div className="setting-group">
              <label className="setting-checkbox">
                <input
                  type="checkbox"
                  checked={settings.showProgress}
                  onChange={(e) => onUpdateSettings({ showProgress: e.target.checked })}
                />
                <span className="checkmark"></span>
                Показувати прогрес виконання
              </label>
            </div>
          </>
        )}

        {/* Вибір інтервалів для тесту розпізнавання */}
        {testType === 'interval_recognition' && (
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
                    checked={settings.intervals.includes(interval.id)}
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
            
            {settings.intervals.length === 0 && (
              <div className="setting-warning">
                <Info className="warning-icon" />
                Оберіть хоча б один інтервал для тестування
              </div>
            )}
          </div>
        )}
      </div>

      {/* Підсумок та початок тесту */}
      <div className="test-start">
        <div className="test-summary">
          <div className="summary-item">
            <Target className="summary-icon" />
            <span>{settings.totalQuestions} питань</span>
          </div>
          {settings.timeLimit && (
            <div className="summary-item">
              <Timer className="summary-icon" />
              <span>{formatTime(settings.timeLimit)}</span>
            </div>
          )}
          {testType === 'interval_recognition' && (
            <div className="summary-item">
              <Star className="summary-icon" />
              <span>{settings.intervals.length} інтервалів</span>
            </div>
          )}
        </div>

        <button
          className="btn btn--primary btn--lg start-test-btn"
          onClick={onStartTest}
          disabled={isLoading || !canStartTest()}
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

        {!canStartTest() && testType === 'interval_recognition' && (
          <div className="setting-info">
            <Info />
            <span>Оберіть хоча б один інтервал для початку тесту</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default TestSettings;