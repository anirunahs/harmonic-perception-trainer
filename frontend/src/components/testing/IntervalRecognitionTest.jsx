import React, { useState, useEffect, useRef } from "react";
import { Play, Square, Volume2, VolumeX, CheckCircle, XCircle, 
         RotateCcw, Headphones, Music, Eye, EyeOff, HelpCircle, 
         Lightbulb, Repeat, SkipForward } from "lucide-react";
import useInstrument from "../../hooks/useInstrument";
import LoadingIndicator from "../LoadingIndicator";

const IntervalRecognitionTest = ({ 
  question, 
  onAnswer, 
  isSubmitting, 
  showCorrectAnswer = false,
  isReviewMode = false,
  showFeedback = true,
  autoNext = false,
  allowRetry = true
}) => {
  const [selectedAnswer, setSelectedAnswer] = useState("");
  const [hasAnswered, setHasAnswered] = useState(false);
  const [showHints, setShowHints] = useState(false);
  const [playCount, setPlayCount] = useState(0);
  const [showNoteNames, setShowNoteNames] = useState(false);
  const [feedbackShown, setFeedbackShown] = useState(false);
  const [confidence, setConfidence] = useState(null);
  const [retryCount, setRetryCount] = useState(0);
  
  // Use instrument hook for better sound quality
  const instrument = useInstrument('piano'); // Can be made configurable
  const [currentPlaying, setCurrentPlaying] = useState(null);
  const [audioError, setAudioError] = useState(null);

  const feedbackTimeoutRef = useRef(null);

  const intervalOptions = [
    { 
      id: "minor_second", 
      name: "Мала секунда", 
      semitones: 1,
      description: "Найменший інтервал, звучить напружено",
      example: "Щелепи (тема з фільму)",
      difficulty: "hard"
    },
    { 
      id: "major_second", 
      name: "Велика секунда", 
      semitones: 2,
      description: "Сусідні ноти, легкий дисонанс",
      example: "До-Ре",
      difficulty: "medium"
    },
    { 
      id: "minor_third", 
      name: "Мала терція", 
      semitones: 3,
      description: "Сумний, меланхолійний звук",
      example: "Початок 'Greensleeves'",
      difficulty: "easy"
    },
    { 
      id: "major_third", 
      name: "Велика терція", 
      semitones: 4,
      description: "Радісний, світлий звук",
      example: "Мажорне тризвуччя",
      difficulty: "easy"
    },
    { 
      id: "perfect_fourth", 
      name: "Чиста кварта", 
      semitones: 5,
      description: "Стабільний, відкритий звук",
      example: "'Here Comes the Bride'",
      difficulty: "easy"
    },
    { 
      id: "tritone", 
      name: "Тритон", 
      semitones: 6,
      description: "Диявольський інтервал, дуже напружений",
      example: "Тема з 'The Simpsons'",
      difficulty: "hard"
    },
    { 
      id: "perfect_fifth", 
      name: "Чиста квінта", 
      semitones: 7,
      description: "Дуже стабільний, порожній звук",
      example: "'Twinkle, Twinkle, Little Star'",
      difficulty: "easy"
    },
    { 
      id: "minor_sixth", 
      name: "Мала секста", 
      semitones: 8,
      description: "Ностальгійний, трохи сумний",
      example: "'The Entertainer'",
      difficulty: "medium"
    },
    { 
      id: "major_sixth", 
      name: "Велика секста", 
      semitones: 9,
      description: "Романтичний, теплий звук",
      example: "'My Bonnie Lies Over the Ocean'",
      difficulty: "medium"
    },
    { 
      id: "minor_seventh", 
      name: "Мала септима", 
      semitones: 10,
      description: "Джазовий, блюзовий характер",
      example: "Домінантсептакорд",
      difficulty: "hard"
    },
    { 
      id: "major_seventh", 
      name: "Велика септима", 
      semitones: 11,
      description: "Дуже напружений, сучасний звук",
      example: "Мажорний септакорд",
      difficulty: "hard"
    },
    { 
      id: "perfect_octave", 
      name: "Чиста октава", 
      semitones: 12,
      description: "Та сама нота, але вище/нижче",
      example: "'Somewhere Over the Rainbow'",
      difficulty: "easy"
    }
  ];

  const [answerOptions, setAnswerOptions] = useState([]);

  useEffect(() => {
    if (question && question.interval_type) {
      const correctInterval = intervalOptions.find(opt => opt.id === question.interval_type);
      const otherIntervals = intervalOptions.filter(opt => opt.id !== question.interval_type);
      
      const numberOfOptions = 4;
      const shuffledOthers = otherIntervals.sort(() => Math.random() - 0.5);
      const selectedOthers = shuffledOthers.slice(0, numberOfOptions - 1);
      
      const options = [correctInterval, ...selectedOthers].sort(() => Math.random() - 0.5);
      setAnswerOptions(options);
    }
  }, [question]);

  useEffect(() => {
    instrument.stopAll();
    setCurrentPlaying(null);
    setSelectedAnswer("");
    setHasAnswered(false);
    setPlayCount(0);
    setFeedbackShown(false);
    setConfidence(null);
    setRetryCount(0);
    
    if (feedbackTimeoutRef.current) {
      clearTimeout(feedbackTimeoutRef.current);
      feedbackTimeoutRef.current = null;
    }
  }, [question, instrument]);

  useEffect(() => {
    if (hasAnswered && showFeedback && !isReviewMode) {
      setFeedbackShown(true);
      
      if (autoNext && feedbackTimeoutRef.current === null) {
        feedbackTimeoutRef.current = setTimeout(() => {
        }, 3000);
      }
    }
  }, [hasAnswered, showFeedback, isReviewMode, autoNext]);

  const handlePlayAudio = async (audioType) => {
    if (!question || !instrument.isLoaded) {
      if (!instrument.isLoaded) {
        setAudioError('Інструмент ще завантажується. Зачекайте...');
      }
      return;
    }
    
    try {
      setAudioError(null);
      
      // Extract base note from question (e.g., "C4" -> "C4")
      const baseNote = question.base_note || 'C4';
      const intervalType = question.interval_type;
      
      const playId = `${question.id}_${audioType}`;
      
      // Stop current playback if same
      if (currentPlaying === playId) {
        instrument.stopAll();
        setCurrentPlaying(null);
        return;
      }
      
      setCurrentPlaying(playId);
      
      if (audioType === 'harmonic') {
        await instrument.playHarmonicInterval(baseNote, intervalType, '2n');
      } else if (audioType === 'melodic') {
        await instrument.playMelodicInterval(baseNote, intervalType, '2n');
      }
      
      setPlayCount(prev => prev + 1);
      
      // Clear playing state after duration
      setTimeout(() => {
        setCurrentPlaying(null);
      }, 2000);
      
    } catch (error) {
      console.error('Помилка відтворення аудіо:', error);
      setAudioError('Не вдалося відтворити аудіо. Спробуйте ще раз.');
      setCurrentPlaying(null);
    }
  };
  
  const handleStopAudio = () => {
    instrument.stopAll();
    setCurrentPlaying(null);
  };

  const handleAnswerSelect = (answerId, confidenceLevel = null) => {
    if (hasAnswered && !isReviewMode && !allowRetry) return;
    
    setSelectedAnswer(answerId);
    setConfidence(confidenceLevel);
    
    if (!isReviewMode) {
      setHasAnswered(true);
      onAnswer(answerId);
    }
  };

  const handleRetry = () => {
    if (!allowRetry || retryCount >= 2) return;
    
    setSelectedAnswer("");
    setHasAnswered(false);
    setFeedbackShown(false);
    setConfidence(null);
    setRetryCount(prev => prev + 1);
  };

  const getOptionClass = (option) => {
    const baseClass = "interval-option";
    
    if (isReviewMode || showCorrectAnswer) {
      if (option.id === question.interval_type) {
        return `${baseClass} ${baseClass}--correct`;
      }
      if (option.id === selectedAnswer && option.id !== question.interval_type) {
        return `${baseClass} ${baseClass}--wrong`;
      }
      if (option.id === selectedAnswer) {
        return `${baseClass} ${baseClass}--selected`;
      }
    } else {
      if (option.id === selectedAnswer) {
        return `${baseClass} ${baseClass}--selected`;
      }
    }
    
    return baseClass;
  };

  const getOptionIcon = (option) => {
    if (!showCorrectAnswer && !isReviewMode) return null;
    
    if (option.id === question.interval_type) {
      return <CheckCircle className="option-icon option-icon--correct" />;
    }
    if (option.id === selectedAnswer && option.id !== question.interval_type) {
      return <XCircle className="option-icon option-icon--wrong" />;
    }
    
    return null;
  };

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case 'easy': return 'text-success';
      case 'medium': return 'text-warning';
      case 'hard': return 'text-error';
      default: return 'text-secondary';
    }
  };

  const getConfidenceButtons = (optionId) => {
    if (hasAnswered || isReviewMode) return null;
    
    return (
      <div className="confidence-buttons">
        <button
          className="confidence-btn confidence-btn--low"
          onClick={() => handleAnswerSelect(optionId, 'low')}
          title="Не впевнений"
        >
          ?
        </button>
        <button
          className="confidence-btn confidence-btn--medium"
          onClick={() => handleAnswerSelect(optionId, 'medium')}
          title="Може бути"
        >
          ~
        </button>
        <button
          className="confidence-btn confidence-btn--high"
          onClick={() => handleAnswerSelect(optionId, 'high')}
          title="Впевнений"
        >
          !
        </button>
      </div>
    );
  };

  if (!question) {
    return (
      <div className="interval-test">
        <div className="interval-test__loading">
          <LoadingIndicator text="Завантаження питання..." />
        </div>
      </div>
    );
  }

  const isCorrect = selectedAnswer === question.interval_type;
  const canRetry = allowRetry && hasAnswered && !isCorrect && retryCount < 2;

  return (
    <div className="interval-test interval-test--enhanced">
      <div className="interval-test__header">
        <div className="question-info">
          <h3 className="interval-test__title">
            <Headphones className="title-icon" />
            Прослухайте інтервал та оберіть правильну відповідь
          </h3>
          
          <div className="test-meta">
            <div className="play-counter">
              <Music className="meta-icon" />
              <span>Прослухано: {playCount} разів</span>
            </div>
            
            {retryCount > 0 && (
              <div className="retry-counter">
                <RotateCcw className="meta-icon" />
                <span>Спроба: {retryCount + 1}/3</span>
              </div>
            )}
          </div>
        </div>

        <div className="question-controls">
          <button
            className={`control-btn ${showNoteNames ? 'control-btn--active' : ''}`}
            onClick={() => setShowNoteNames(!showNoteNames)}
            title={showNoteNames ? "Приховати назви нот" : "Показати назви нот"}
          >
            {showNoteNames ? <EyeOff /> : <Eye />}
          </button>
          
          <button
            className={`control-btn ${showHints ? 'control-btn--active' : ''}`}
            onClick={() => setShowHints(!showHints)}
            title={showHints ? "Приховати підказки" : "Показати підказки"}
          >
            {showHints ? <Lightbulb className="active" /> : <HelpCircle />}
          </button>
        </div>
      </div>
      
      <div className="interval-test__info">
        <div className="note-display">
          <div className="note-item">
            <span className="note-label">Базова нота:</span>
            <span className="note-value">{question.base_note}</span>
          </div>
          {(showNoteNames || isReviewMode) && (
            <div className="note-item">
              <span className="note-label">Цільова нота:</span>
              <span className="note-value">{question.target_note}</span>
            </div>
          )}
        </div>

        {showHints && !hasAnswered && (
          <div className="hints-panel">
            <h4> Підказки для розпізнавання:</h4>
            <ul>
              <li>Прослухайте інтервал кілька разів</li>
              <li>Спочатку спробуйте гармонічне звучання, потім мелодичне</li>
              <li>Зверніть увагу на характер звучання (стабільний/напружений)</li>
              <li>Порівняйте з відомими вам мелодіями</li>
            </ul>
          </div>
        )}
      </div>

      <div className="interval-test__audio">
        <div className="audio-controls audio-controls--enhanced">
          <button
            className={`audio-btn audio-btn--harmonic ${
              currentPlaying === `${question.id}_harmonic` ? 'audio-btn--playing' : ''
            }`}
            onClick={() => handlePlayAudio('harmonic')}
            disabled={!instrument.isLoaded || instrument.isLoading}
          >
            {instrument.isLoading ? (
              <LoadingIndicator size="small" />
            ) : currentPlaying === `${question.id}_harmonic` ? (
              <Square />
            ) : (
              <Volume2 />
            )}
            <div className="audio-btn__content">
              <span className="audio-btn__label">Гармонічно</span>
              <span className="audio-btn__description">Ноти разом</span>
            </div>
          </button>

          <button
            className={`audio-btn audio-btn--melodic ${
              currentPlaying === `${question.id}_melodic` ? 'audio-btn--playing' : ''
            }`}
            onClick={() => handlePlayAudio('melodic')}
            disabled={!instrument.isLoaded || instrument.isLoading}
          >
            {instrument.isLoading ? (
              <LoadingIndicator size="small" />
            ) : currentPlaying === `${question.id}_melodic` ? (
              <Square />
            ) : (
              <Play />
            )}
            <div className="audio-btn__content">
              <span className="audio-btn__label">Мелодично</span>
              <span className="audio-btn__description">Ноти поступово</span>
            </div>
          </button>

          <button
            className="audio-btn audio-btn--repeat"
            onClick={() => handlePlayAudio(currentPlaying?.includes('harmonic') ? 'harmonic' : 'melodic')}
            disabled={!currentPlaying && playCount === 0}
            title="Повторити останнє відтворення"
          >
            <Repeat />
            <span>Повторити</span>
          </button>
        </div>

        {currentPlaying && (
          <button
            className="audio-btn audio-btn--stop"
            onClick={handleStopAudio}
          >
            <Square />
            <span>Зупинити</span>
          </button>
        )}

        {audioError && (
          <div className="audio-error">
            <VolumeX />
            <span>{audioError}</span>
            <button onClick={() => setAudioError(null)}>×</button>
          </div>
        )}
      </div>

      <div className="interval-test__options">
        <h4 className="options-title">Оберіть інтервал:</h4>
        
        <div className="options-grid options-grid--enhanced">
          {answerOptions.map((option) => (
            <div key={option.id} className="option-container">
              <button
                className={getOptionClass(option)}
                onClick={() => handleAnswerSelect(option.id)}
                disabled={isSubmitting || (hasAnswered && !isReviewMode && !allowRetry)}
              >
                <div className="option-content">
                  <div className="option-header">
                    <span className="option-name">{option.name}</span>
                    <span className={`option-difficulty ${getDifficultyColor(option.difficulty)}`}>
                      {option.difficulty === 'easy' ? 'Легкий' : 
                       option.difficulty === 'medium' ? 'Середній' : 'Складний'}
                    </span>
                  </div>
                  <span className="option-semitones">{option.semitones} півтонів</span>
                  
                  {showHints && (
                    <div className="option-hints">
                      <p className="option-description">{option.description}</p>
                      <p className="option-example">Приклад: {option.example}</p>
                    </div>
                  )}
                </div>
                {getOptionIcon(option)}
              </button>
              
              {!hasAnswered && !isReviewMode && (
                <div className="confidence-selector">
                  <span className="confidence-label">Впевненість:</span>
                  {getConfidenceButtons(option.id)}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {showFeedback && hasAnswered && (
        <div className="interval-test__feedback">
          <div className={`feedback feedback--${isCorrect ? 'correct' : 'wrong'}`}>
            <div className="feedback-content">
              {isCorrect ? (
                <>
                  <CheckCircle className="feedback-icon" />
                  <div className="feedback-text">
                    <h4>Правильно!</h4>
                    <p>Це дійсно {intervalOptions.find(opt => opt.id === question.interval_type)?.name}</p>
                    {confidence && (
                      <p className="confidence-feedback">
                        Рівень впевненості: {
                          confidence === 'high' ? 'Високий' :
                          confidence === 'medium' ? 'Середній' : 'Низький'
                        }
                      </p>
                    )}
                  </div>
                </>
              ) : (
                <>
                  <XCircle className="feedback-icon" />
                  <div className="feedback-text">
                    <h4>Неправильно</h4>
                    <p>
                      Правильна відповідь: <strong>{intervalOptions.find(opt => opt.id === question.interval_type)?.name}</strong>
                    </p>
                    {selectedAnswer && (
                      <p className="user-answer">
                        Ваша відповідь: {intervalOptions.find(opt => opt.id === selectedAnswer)?.name}
                      </p>
                    )}
                    {canRetry && (
                      <div className="retry-section">
                        <p>У вас є ще {2 - retryCount} спроби</p>
                        <button 
                          className="btn btn--ghost btn--sm"
                          onClick={handleRetry}
                        >
                          <RotateCcw />
                          Спробувати ще раз
                        </button>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
            
            {(isCorrect || !canRetry) && (
              <div className="interval-info-panel">
                {(() => {
                  const correctInterval = intervalOptions.find(opt => opt.id === question.interval_type);
                  return (
                    <div className="interval-details">
                      <h5> Про цей інтервал:</h5>
                      <p><strong>Характер:</strong> {correctInterval?.description}</p>
                      <p><strong>Приклад:</strong> {correctInterval?.example}</p>
                      <p><strong>Відстань:</strong> {correctInterval?.semitones} півтонів</p>
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
        </div>
      )}

      {autoNext && hasAnswered && (
        <div className="auto-next-indicator">
          <SkipForward className="auto-next-icon" />
          <span>Автоматичний перехід через 3 секунди...</span>
        </div>
      )}
    </div>
  );
};

export default IntervalRecognitionTest;