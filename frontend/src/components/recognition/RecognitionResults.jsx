import React from 'react';
import { TrendingUp, Clock, Volume2, AlertCircle, Info, CheckCircle, AlertTriangle } from 'lucide-react';

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
          <p>Обробка аудіо та розпізнавання інтервалу</p>
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

          {result.alternative_predictions && result.alternative_predictions.length > 0 && (
            <div className="alternative-results">
              <h3>Альтернативні варіанти:</h3>
              <div className="alternatives-list">
                {result.alternative_predictions.map((pred, index) => (
                  <div key={index} className="alternative-item">
                    <div className="alternative-item__info">
                      <span className="alternative-item__name">
                        {getIntervalDisplayName(pred.interval)}
                      </span>
                      <span className="alternative-item__subtitle">
                        {pred.interval}
                      </span>
                    </div>
                    <div className="alternative-item__stats">
                      <span className="alternative-item__confidence">
                        {(pred.confidence * 100).toFixed(1)}%
                      </span>
                      {pred.occurrence_count && (
                        <span className="alternative-item__count">
                          ({pred.occurrence_count} сегм.)
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.recommendations && result.recommendations.length > 0 && (
            <div className="recommendations">
              <h3>Рекомендації для покращення:</h3>
              <div className="recommendations-list">
                {result.recommendations.map((rec, index) => (
                  <div key={index} className={`recommendation recommendation--${rec.severity}`}>
                    {getSeverityIcon(rec.severity)}
                    <span>{rec.message}</span>
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