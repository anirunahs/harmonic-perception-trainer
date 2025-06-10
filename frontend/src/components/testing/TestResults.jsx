import React from "react";
import { ChevronLeft, RotateCcw, CheckCircle, Trophy, Target, 
         Award, Timer, TrendingUp, Brain, Star } from "lucide-react";

const TestResults = ({ 
  results, 
  testTypes, 
  answers = {}, 
  intervals = [],
  timeElapsed = 0,
  streakCount = 0,
  longestStreak = 0,
  onRetry, 
  onReturnToMenu 
}) => {
  // Визначення рівня результату
  const getPerformanceLevel = (accuracy) => {
    if (accuracy >= 90) return { 
      level: "Відмінно", 
      color: "text-success", 
      icon: Trophy, 
      gradient: "from-yellow-400 to-orange-500",
      description: "Феноменальний результат! Ви демонструєте глибоке розуміння музичних інтервалів."
    };
    if (accuracy >= 80) return { 
      level: "Добре", 
      color: "text-primary", 
      icon: Star,
      gradient: "from-blue-400 to-purple-500", 
      description: "Чудовий результат! Ви на правильному шляху до майстерності."
    };
    if (accuracy >= 70) return { 
      level: "Задовільно", 
      color: "text-warning", 
      icon: Target,
      gradient: "from-green-400 to-blue-500",
      description: "Непоганий результат! Продовжуйте тренуватися для покращення."
    };
    if (accuracy >= 50) return { 
      level: "Потребує покращення", 
      color: "text-orange-500", 
      icon: TrendingUp,
      gradient: "from-orange-400 to-red-500",
      description: "Є простір для зростання. Рекомендуємо додаткове тренування."
    };
    return { 
      level: "Потребує значного покращення", 
      color: "text-error", 
      icon: Brain,
      gradient: "from-red-400 to-pink-500",
      description: "Не засмучуйтесь! Кожен починає з перших кроків. Продовжуйте практикуватися."
    };
  };

  // Розрахунок додаткових метрик
  const calculateAdvancedMetrics = () => {
    const answeredQuestions = Object.values(answers);
    
    // Середній час відповіді
    const responseTimes = answeredQuestions
      .filter(a => a.response_time)
      .map(a => a.response_time);
    
    const avgResponseTime = responseTimes.length > 0 
      ? responseTimes.reduce((sum, time) => sum + time, 0) / responseTimes.length 
      : 0;

    // Розподіл за складністю
    const difficultyStats = {};
    intervals.forEach(interval => {
      const difficulty = getDifficultyLevel(interval.id);
      if (!difficultyStats[difficulty]) {
        difficultyStats[difficulty] = { correct: 0, total: 0 };
      }
      
      const intervalAnswers = answeredQuestions.filter(a => 
        a.question?.interval_type === interval.id
      );
      
      difficultyStats[difficulty].total += intervalAnswers.length;
      difficultyStats[difficulty].correct += intervalAnswers.filter(a => a.is_correct).length;
    });

    return {
      avgResponseTime: Math.round(avgResponseTime * 10) / 10,
      fastestResponse: responseTimes.length > 0 ? Math.min(...responseTimes) : 0,
      slowestResponse: responseTimes.length > 0 ? Math.max(...responseTimes) : 0,
      difficultyStats
    };
  };

  // Визначення складності інтервалу
  const getDifficultyLevel = (intervalId) => {
    const easyIntervals = ['minor_third', 'major_third', 'perfect_fourth', 'perfect_fifth', 'perfect_octave'];
    const mediumIntervals = ['major_second', 'minor_sixth', 'major_sixth'];
    
    if (easyIntervals.includes(intervalId)) return 'easy';
    if (mediumIntervals.includes(intervalId)) return 'medium';
    return 'hard';
  };

  // Форматування часу
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Отримання назви інтервалу
  const getIntervalName = (intervalId) => {
    const intervalNames = {
      minor_second: "Мала секунда",
      major_second: "Велика секунда", 
      minor_third: "Мала терція",
      major_third: "Велика терція",
      perfect_fourth: "Чиста кварта",
      tritone: "Тритон",
      perfect_fifth: "Чиста квінта",
      minor_sixth: "Мала секста",
      major_sixth: "Велика секста",
      minor_seventh: "Мала септима",
      major_seventh: "Велика септима",
      perfect_octave: "Чиста октава"
    };
    return intervalNames[intervalId] || intervalId;
  };

  if (!results) return null;

  const performance = getPerformanceLevel(results.accuracy_percentage);
  const PerformanceIcon = performance.icon;
  const metrics = calculateAdvancedMetrics();
  const testTypeName = testTypes?.find(t => t.id === results.test_type)?.name || "Тест";

  return (
    <div className="test-results">
      <div className="test-results__header">
        <div className="results-performance">
          <div className={`performance-icon bg-gradient-to-r ${performance.gradient} p-4 rounded-full`}>
            <PerformanceIcon className="w-8 h-8 text-white" />
          </div>
          <div className="performance-text">
            <h1 className={`results-title ${performance.color}`}>
              {performance.level}
            </h1>
            <p className="results-description">
              {performance.description}
            </p>
          </div>
        </div>
        
        <div className="results-meta">
          <p className="results-subtitle">{testTypeName}</p>
          <div className="results-score">
            <span className="score-value">{results.accuracy_percentage.toFixed(1)}%</span>
            <span className="score-label">Точність</span>
          </div>
        </div>
      </div>

      <div className="test-results__stats">
        <div className="stats-grid">
          <div className="stat-card stat-card--primary">
            <div className="stat-icon">
              <CheckCircle className="w-6 h-6 text-success" />
            </div>
            <div className="stat-content">
              <div className="stat-value">{results.correct_answers}</div>
              <div className="stat-label">Правильних відповідей</div>
              <div className="stat-sublabel">з {results.total_questions}</div>
            </div>
          </div>
          
          <div className="stat-card stat-card--secondary">
            <div className="stat-icon">
              <Target className="w-6 h-6 text-primary" />
            </div>
            <div className="stat-content">
              <div className="stat-value">{results.accuracy_percentage.toFixed(1)}%</div>
              <div className="stat-label">Загальна точність</div>
              <div className="stat-sublabel">середній показник</div>
            </div>
          </div>
          
          <div className="stat-card stat-card--accent">
            <div className="stat-icon">
              <Award className="w-6 h-6 text-warning" />
            </div>
            <div className="stat-content">
              <div className="stat-value">+{results.experience_gained}</div>
              <div className="stat-label">Досвід отримано</div>
              <div className="stat-sublabel">XP бонус</div>
            </div>
          </div>
          
          {timeElapsed > 0 && (
            <div className="stat-card stat-card--info">
              <div className="stat-icon">
                <Timer className="w-6 h-6 text-secondary" />
              </div>
              <div className="stat-content">
                <div className="stat-value">{formatTime(timeElapsed)}</div>
                <div className="stat-label">Загальний час</div>
                <div className="stat-sublabel">витрачено</div>
              </div>
            </div>
          )}
        </div>

        <div className="accuracy-visualization">
          <div className="accuracy-bar">
            <div 
              className="accuracy-bar__fill" 
              style={{ 
                width: `${results.accuracy_percentage}%`,
                background: results.accuracy_percentage >= 80 
                  ? 'linear-gradient(90deg, #10b981, #059669)' 
                  : results.accuracy_percentage >= 70 
                  ? 'linear-gradient(90deg, #f59e0b, #d97706)'
                  : 'linear-gradient(90deg, #ef4444, #dc2626)'
              }} 
            />
            <span className="accuracy-bar__text">
              {results.accuracy_percentage.toFixed(1)}% точність
            </span>
          </div>
        </div>

        {metrics.avgResponseTime > 0 && (
          <div className="advanced-metrics">
            <h3 className="metrics-title">Детальна статистика</h3>
            <div className="metrics-grid">
              <div className="metric-item">
                <span className="metric-label">Середній час відповіді:</span>
                <span className="metric-value">{formatTime(metrics.avgResponseTime)}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Найшвидша відповідь:</span>
                <span className="metric-value">{formatTime(metrics.fastestResponse)}</span>
              </div>
              {longestStreak > 0 && (
                <div className="metric-item">
                  <span className="metric-label">Найдовша серія:</span>
                  <span className="metric-value">{longestStreak} поспіль</span>
                </div>
              )}
            </div>
          </div>
        )}

        {results.test_type === 'interval_recognition' && Object.keys(answers).length > 0 && (
          <div className="detailed-results">
            <h3 className="details-title">Результати по інтервалах</h3>
            <div className="interval-results">
              {Object.entries(
                Object.values(answers).reduce((acc, answer) => {
                  if (!answer.question?.interval_type) return acc;
                  
                  const intervalName = getIntervalName(answer.question.interval_type);
                  if (!acc[intervalName]) {
                    acc[intervalName] = { correct: 0, total: 0, difficulty: getDifficultyLevel(answer.question.interval_type) };
                  }
                  acc[intervalName].total++;
                  if (answer.is_correct) acc[intervalName].correct++;
                  return acc;
                }, {})
              ).map(([intervalName, stats]) => {
                const percentage = Math.round((stats.correct / stats.total) * 100);
                const isGood = percentage >= 70;
                
                return (
                  <div key={intervalName} className="interval-result-item">
                    <div className="interval-result-header">
                      <span className="interval-name">{intervalName}</span>
                      <span className={`interval-difficulty difficulty--${stats.difficulty}`}>
                        {stats.difficulty === 'easy' ? 'Легкий' : 
                         stats.difficulty === 'medium' ? 'Середній' : 'Складний'}
                      </span>
                    </div>
                    <div className="interval-result-stats">
                      <span className={`interval-score ${isGood ? 'text-success' : 'text-warning'}`}>
                        {stats.correct}/{stats.total} ({percentage}%)
                      </span>
                      <div className="interval-progress">
                        <div 
                          className="interval-progress__fill"
                          style={{ 
                            width: `${percentage}%`,
                            backgroundColor: isGood ? '#10b981' : '#f59e0b'
                          }}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        <div className="recommendations">
          <h3 className="recommendations-title">Рекомендації для покращення</h3>
          <div className="recommendations-list">
            {results.accuracy_percentage < 70 && (
              <div className="recommendation-item">
                <span className="recommendation-icon">📚</span>
                <span>Рекомендуємо додатково потренуватися з базовими інтервалами</span>
              </div>
            )}
            {metrics.avgResponseTime > 15 && (
              <div className="recommendation-item">
                <span className="recommendation-icon">⚡</span>
                <span>Спробуйте швидше розпізнавати інтервали для покращення реакції</span>
              </div>
            )}
            {longestStreak < 3 && results.total_questions > 5 && (
              <div className="recommendation-item">
                <span className="recommendation-icon">🎯</span>
                <span>Зосередьтеся на послідовності - намагайтеся відповідати правильно декілька разів поспіль</span>
              </div>
            )}
            {results.accuracy_percentage >= 90 && (
              <div className="recommendation-item">
                <span className="recommendation-icon">🏆</span>
                <span>Чудовий результат! Спробуйте більш складні інтервали або збільште швидкість</span>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="test-results__actions">
        <button
          className="btn btn--primary btn--lg"
          onClick={onRetry}
        >
          <RotateCcw className="w-5 h-5" />
          <span>Пройти ще раз</span>
        </button>

        <button
          className="btn btn--ghost btn--lg"
          onClick={onReturnToMenu}
        >
          <ChevronLeft className="w-5 h-5" />
          <span>Назад до меню</span>
        </button>
      </div>
    </div>
  );
};

export default TestResults;