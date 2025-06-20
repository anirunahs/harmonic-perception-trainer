import React from 'react';
import { TrendingUp, Clock, Volume2, AlertCircle, Info, CheckCircle, AlertTriangle, Music, Target, Award } from 'lucide-react';

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
          <p>Обробка аудіо та розпізнавання музичних інтервалів</p>
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
            <h3>Помилка аналізу</h3>
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

  const getQualityText = (score) => {
    if (score >= 0.8) return 'Відмінна';
    if (score >= 0.6) return 'Хороша';
    if (score >= 0.4) return 'Задовільна';
    return 'Низька';
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

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'error':
        return <AlertCircle className="severity-icon severity-icon--error" />;
      case 'warning':
        return <AlertTriangle className="severity-icon severity-icon--warning" />;
      case 'info':
      default:
        return <Info className="severity-icon severity-icon--info" />;
    }
  };

  if (result.status === 'error') {
    return (
      <div className="results-panel">
        <h2 className="results-panel__title">Результат аналізу</h2>
        <div className="analysis-error">
          <AlertCircle />
          <div>
            <h3>Не вдалося розпізнати інтервал</h3>
            <p>{result.message}</p>
            {result.warnings && result.warnings.length > 0 && (
              <div className="warnings-list">
                <h4>Попередження:</h4>
                <ul>
                  {result.warnings.map((warning, index) => (
                    <li key={index}>{warning}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
        {result.processing_info && (
          <div className="processing-info">
            <div className="info-grid">
              <div className="info-item">
                <span className="info-label">Час обробки:</span>
                <span className="info-value">
                  {result.processing_info.processing_time?.toFixed(2)}с
                </span>
              </div>
              {result.processing_info.quality_score && (
                <div className="info-item">
                  <span className="info-label">Якість запису:</span>
                  <span className="info-value">
                    {(result.processing_info.quality_score * 100).toFixed(0)}%
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="results-panel">
      <h2 className="results-panel__title">
        <TrendingUp />
        Результат розпізнавання
        {result.mode && (
          <span className="results-panel__mode">({result.mode})</span>
        )}
      </h2>
      
      {result.status === 'success' && result.best_prediction && (
        <div className="recognition-success">
          <div className="main-result">
            <div className="interval-display">
              <div className="interval-display__header">
                <Award className="main-result-icon" />
                <span className="main-result-label">Найкращий результат</span>
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
                      width: `${result.best_prediction.confidence * 100}%`,
                      backgroundColor: getConfidenceColor(result.best_prediction.confidence)
                    }}
                  />
                </div>
                <span className="confidence-text">
                  Впевненість: {(result.best_prediction.confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>

          {result.segments_analysis && result.segments_analysis.length > 0 && (
            <div className="segments-analysis">
              <h3 className="segments-analysis__title">
                <Volume2 />
                Аналіз по сегментах
                <span className="segments-count">({result.segments_analysis.length} сегм.)</span>
              </h3>
              
              <div className="segments-grid">
                {result.segments_analysis.map((segment, index) => (
                  <div key={segment.segment_id} className="segment-card">
                    <div className="segment-card__header">
                      <div className="segment-card__info">
                        <h4 className="segment-card__title">
                          Сегмент {segment.segment_id}
                        </h4>
                        <span className="segment-card__time">
                          {segment.time_range} ({segment.duration})
                        </span>
                      </div>
                      <div className="segment-card__quality">
                        <div 
                          className="quality-indicator-small"
                          style={{ 
                            backgroundColor: getConfidenceColor(segment.audio_quality)
                          }}
                        />
                        <span className="quality-text-small">
                          {(segment.audio_quality * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>

                    <div className="segment-card__best">
                      <div className="best-prediction">
                        <Award className="best-prediction__icon" />
                        <div className="best-prediction__content">
                          <div className="best-prediction__name">
                            {getIntervalDisplayName(segment.best_interval)}
                          </div>
                          <div className="best-prediction__confidence">
                            {(segment.best_confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="segment-card__predictions">
                      <h5 className="predictions-title">ТОП-3 варіанти:</h5>
                      <div className="predictions-list">
                        {segment.top_predictions.slice(0, 3).map((prediction, predIndex) => (
                          <div 
                            key={predIndex} 
                            className={`prediction-item ${predIndex === 0 ? 'prediction-item--best' : ''}`}
                          >
                            <div className="prediction-item__rank">
                              {getRankIcon(predIndex + 1)}
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
                                  width: `${prediction.confidence * 100}%`,
                                  backgroundColor: getConfidenceColor(prediction.confidence)
                                }}
                              />
                              <span className="confidence-percentage">
                                {(prediction.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.overall_top_intervals && result.overall_top_intervals.length > 0 && (
            <div className="overall-top-intervals">
              <h3 className="overall-top-intervals__title">
                <Target />
                Загальний рейтинг інтервалів
              </h3>
              <div className="overall-intervals-list">
                {result.overall_top_intervals.map((interval, index) => (
                  <div key={index} className="overall-interval-item">
                    <div className="overall-interval-item__rank">
                      {getRankIcon(index + 1)}
                    </div>
                    <div className="overall-interval-item__content">
                      <div className="overall-interval-item__name">
                        {getIntervalDisplayName(interval.interval)}
                      </div>
                      <div className="overall-interval-item__subtitle">
                        {interval.interval}
                      </div>
                      <div className="overall-interval-item__stats">
                        Знайдено в {interval.segments_found_in} сегм.
                      </div>
                    </div>
                    <div className="overall-interval-item__confidence">
                      <div 
                        className="confidence-bar-large"
                        style={{
                          width: `${interval.confidence * 100}%`,
                          backgroundColor: getConfidenceColor(interval.confidence)
                        }}
                      />
                      <span className="confidence-percentage-large">
                        {(interval.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.mode === 'quick' && result.top_predictions && (
            <div className="quick-mode-results">
              <h3 className="quick-mode-results__title">
                <Target />
                ТОП-3 варіанти (швидкий режим)
              </h3>
              <div className="quick-predictions-list">
                {result.top_predictions.map((prediction, index) => (
                  <div key={index} className="quick-prediction-item">
                    <div className="quick-prediction-item__rank">
                      {getRankIcon(index + 1)}
                    </div>
                    <div className="quick-prediction-item__content">
                      <div className="quick-prediction-item__name">
                        {getIntervalDisplayName(prediction.interval)}
                      </div>
                      <div className="quick-prediction-item__subtitle">
                        {prediction.interval}
                      </div>
                    </div>
                    <div className="quick-prediction-item__confidence">
                      <div 
                        className="confidence-bar-large"
                        style={{
                          width: `${prediction.confidence * 100}%`,
                          backgroundColor: getConfidenceColor(prediction.confidence)
                        }}
                      />
                      <span className="confidence-percentage-large">
                        {(prediction.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.processing_info && (
            <div className="processing-info">
              <h3>Деталі аналізу:</h3>
              <div className="info-grid">
                {result.segments_analyzed && (
                  <div className="info-item">
                    <Volume2 className="info-icon" />
                    <div className="info-content">
                      <span className="info-label">Сегментів проаналізовано:</span>
                      <span className="info-value">{result.segments_analyzed}</span>
                    </div>
                  </div>
                )}
                
                <div className="info-item">
                  <Clock className="info-icon" />
                  <div className="info-content">
                    <span className="info-label">Час обробки:</span>
                    <span className="info-value">
                      {result.processing_info.processing_time?.toFixed(2)}с
                    </span>
                  </div>
                </div>

                {result.processing_info.quality_score !== undefined && (
                  <div className="info-item">
                    <CheckCircle className="info-icon" />
                    <div className="info-content">
                      <span className="info-label">Якість запису:</span>
                      <span className="info-value">
                        {getQualityText(result.processing_info.quality_score)} 
                        ({(result.processing_info.quality_score * 100).toFixed(0)}%)
                      </span>
                    </div>
                  </div>
                )}

                {result.processing_info.original_duration && (
                  <div className="info-item">
                    <Clock className="info-icon" />
                    <div className="info-content">
                      <span className="info-label">Тривалість запису:</span>
                      <span className="info-value">
                        {result.processing_info.original_duration.toFixed(1)}с
                      </span>
                    </div>
                  </div>
                )}

                {result.processing_info.preprocessing_level && (
                  <div className="info-item">
                    <TrendingUp className="info-icon" />
                    <div className="info-content">
                      <span className="info-label">Рівень обробки:</span>
                      <span className="info-value">
                        {result.processing_info.preprocessing_level}
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {result.processing_info.warnings && result.processing_info.warnings.length > 0 && (
                <div className="warnings-section">
                  <h4>Попередження:</h4>
                  <ul className="warnings-list">
                    {result.processing_info.warnings.map((warning, index) => (
                      <li key={index} className="warning-item">
                        <AlertTriangle className="warning-icon" />
                        {warning}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {result.detailed_results && process.env.NODE_ENV === 'development' && (
            <details className="detailed-results">
              <summary>Детальні результати (тест)</summary>
              <pre className="detailed-json">
                {JSON.stringify(result.detailed_results, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
};

export default RecognitionResults;