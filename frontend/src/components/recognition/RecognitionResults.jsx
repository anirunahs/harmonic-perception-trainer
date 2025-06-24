import React from 'react';
import { TrendingUp, Clock, AlertCircle, Award, Target, Music } from 'lucide-react';

const IntervalNames = {
  'minor_2nd': 'Мала секунда',
  'major_2nd': 'Велика секунда', 
  'minor_3rd': 'Мала терція',
  'major_3rd': 'Велика терція',
  'perfect_4th': 'Чиста кварта',
  'tritone': 'Тритон',
  'perfect_5th': 'Чиста квінта',
  'minor_6th': 'Мала секста',
  'major_6th': 'Велика секста',
  'minor_7th': 'Мала септима',
  'major_7th': 'Велика септима',
  'perfect_8th': 'Чиста октава'
};

const RecognitionResults = ({ result, isAnalyzing, error }) => {
  if (isAnalyzing) {
    return (
      <div className="results-panel">
        <div className="analysis-loading">
          <div className="loading-spinner">
            <div className="spinner"></div>
          </div>
          <h2>Аналізуємо запис...</h2>
          <p>Обробка аудіо та розпізнавання музичних характеристик</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="results-panel">
        <div className="analysis-error">
          <AlertCircle />
          <div>
            <h3>Помилка розпізнавання</h3>
            <p>{error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!result) {
    return null;
  }

  const getIntervalDisplayName = (interval) => {
    return IntervalNames[interval] || interval;
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#10b981';
    if (confidence >= 0.6) return '#f59e0b';
    return '#ef4444';
  };

  const getRankIcon = (rank) => {
    switch (rank) {
      case 1:
        return <Award className="rank-icon rank-icon--gold" />;
      case 2:
        return <Target className="rank-icon rank-icon--silver" />;
      case 3:
        return <Music className="rank-icon rank-icon--bronze" />;
      default:
        return <div className="rank-number">{rank}</div>;
    }
  };

  if (result.status === 'error') {
    return (
      <div className="results-panel">
        <h2 className="results-panel__title">Результат розпізнавання</h2>
        <div className="analysis-error">
          <AlertCircle />
          <div>
            <h3>Не вдалося розпізнати інтервал</h3>
            <p>{result.message || result.error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="results-panel">
      <h2 className="results-panel__title">
        <TrendingUp />
        Результат розпізнавання
      </h2>
      
      {result.status === 'success' && result.best_prediction && (
        <div className="recognition-success">
          <div className="main-result">
            <div className="interval-display">
              <div className="interval-display__header">
                <Award className="main-result-icon" />
                <span className="main-result-label">Розпізнаний інтервал</span>
              </div>
              <div className="interval-display__name">
                {getIntervalDisplayName(result.best_prediction.interval)}
              </div>
              <div className="interval-display__subtitle">
                {result.best_prediction.interval}
              </div>
              <div className="interval-display__confidence">
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill"
                    style={{ 
                      width: `${result.best_prediction.percentage}%`,
                      backgroundColor: getConfidenceColor(result.best_prediction.confidence)
                    }}
                  />
                </div>
                <span className="confidence-text">
                  Впевненість: {result.best_prediction.percentage}%
                </span>
              </div>
            </div>
          </div>

          {result.top_predictions && result.top_predictions.length > 1 && (
            <div className="top-predictions">
              <h3 className="top-predictions__title">
                <Target />
                Альтернативні варіанти
              </h3>
              <div className="predictions-list">
                {result.top_predictions.slice(1).map((prediction, index) => (
                  <div key={index} className="prediction-item">
                    <div className="prediction-item__rank">
                      {getRankIcon(prediction.rank)}
                    </div>
                    <div className="prediction-item__content">
                      <div className="prediction-item__name">
                        {getIntervalDisplayName(prediction.interval)}
                      </div>
                      <div className="prediction-item__subtitle">
                        {prediction.interval}
                      </div>
                    </div>
                    <div className="prediction-item__confidence">
                      <div 
                        className="confidence-mini-bar"
                        style={{
                          width: `${prediction.percentage}%`,
                          backgroundColor: getConfidenceColor(prediction.confidence)
                        }}
                      />
                      <span className="confidence-percentage">
                        {prediction.percentage}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="processing-info">
            <h3>Деталі аналізу:</h3>
            <div className="info-grid">
              <div className="info-item">
                <Clock className="info-icon" />
                <div className="info-content">
                  <span className="info-label">Час обробки:</span>
                  <span className="info-value">{result.processing_time}с</span>
                </div>
              </div>

              {result.audio_duration && (
                <div className="info-item">
                  <Clock className="info-icon" />
                  <div className="info-content">
                    <span className="info-label">Тривалість запису:</span>
                    <span className="info-value">{result.audio_duration}с</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RecognitionResults;