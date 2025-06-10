import React, { useState, useEffect } from "react";
import { Play, Square, Volume2, VolumeX, CheckCircle, XCircle } from "lucide-react";
import { useAudioPlayer } from "../../hooks/useAudioPlayer";
import LoadingIndicator from "../LoadingIndicator";

const IntervalRecognitionTest = ({ 
  question, 
  onAnswer, 
  isSubmitting, 
  showCorrectAnswer = false,
  isReviewMode = false 
}) => {
  const [selectedAnswer, setSelectedAnswer] = useState("");
  const [hasAnswered, setHasAnswered] = useState(false);
  
  const {
    currentPlaying,
    loadingAudio,
    audioError,
    setAudioError,
    playAudio,
    stopAudio,
    clearAllAudio
  } = useAudioPlayer();

  const intervalOptions = [
    { id: "minor_second", name: "Мала секунда", semitones: 1 },
    { id: "major_second", name: "Велика секунда", semitones: 2 },
    { id: "minor_third", name: "Мала терція", semitones: 3 },
    { id: "major_third", name: "Велика терція", semitones: 4 },
    { id: "perfect_fourth", name: "Чиста кварта", semitones: 5 },
    { id: "tritone", name: "Тритон", semitones: 6 },
    { id: "perfect_fifth", name: "Чиста квінта", semitones: 7 },
    { id: "minor_sixth", name: "Мала секста", semitones: 8 },
    { id: "major_sixth", name: "Велика секста", semitones: 9 },
    { id: "minor_seventh", name: "Мала септима", semitones: 10 },
    { id: "major_seventh", name: "Велика септима", semitones: 11 },
    { id: "perfect_octave", name: "Чиста октава", semitones: 12 }
  ];

  const [answerOptions, setAnswerOptions] = useState([]);

  useEffect(() => {
    if (question && question.interval_type) {
      const correctInterval = intervalOptions.find(opt => opt.id === question.interval_type);
      const otherIntervals = intervalOptions.filter(opt => opt.id !== question.interval_type);
      
      const shuffledOthers = otherIntervals.sort(() => Math.random() - 0.5);
      const selectedOthers = shuffledOthers.slice(0, 2);
      
      const options = [correctInterval, ...selectedOthers].sort(() => Math.random() - 0.5);
      setAnswerOptions(options);
    }
  }, [question]);

  useEffect(() => {
    clearAllAudio();
    setSelectedAnswer("");
    setHasAnswered(false);
  }, [question, clearAllAudio]);

  const handlePlayAudio = (audioType) => {
    if (!question) return;
    
    const fakeIntervals = [{
      id: question.id,
      harmonic_url: question.harmonic_audio_url,
      melodic_url: question.melodic_audio_url
    }];
    
    playAudio(question.id, audioType, fakeIntervals);
  };

  const handleAnswerSelect = (answerId) => {
    if (hasAnswered && !isReviewMode) return;
    
    setSelectedAnswer(answerId);
    
    if (!isReviewMode) {
      setHasAnswered(true);
      onAnswer(answerId);
    }
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

  if (!question) {
    return (
      <div className="interval-test">
        <div className="interval-test__loading">
          <LoadingIndicator text="Завантаження питання..." />
        </div>
      </div>
    );
  }

  return (
    <div className="interval-test">
      <div className="interval-test__question">
        <h3 className="interval-test__title">
          Прослухайте інтервал та оберіть правильну відповідь
        </h3>
        
        <div className="interval-test__info">
          <span className="note-info">Базова нота: <strong>{question.base_note}</strong></span>
          {isReviewMode && (
            <span className="note-info">Цільова нота: <strong>{question.target_note}</strong></span>
          )}
        </div>
      </div>

      <div className="interval-test__audio">
        <div className="audio-controls">
          <button
            className={`audio-btn audio-btn--harmonic ${
              currentPlaying === `${question.id}_harmonic` ? 'audio-btn--playing' : ''
            }`}
            //onClick={() => handlePlayAudio('harmonic')}
            disabled={loadingAudio === `${question.id}_harmonic`}
          >
            {loadingAudio === `${question.id}_harmonic` ? (
              <LoadingIndicator size="small" />
            ) : currentPlaying === `${question.id}_harmonic` ? (
              <Square />
            ) : (
              <Volume2 />
            )}
            <span>Одночасно</span>
          </button>

          <button
            className={`audio-btn audio-btn--melodic ${
              currentPlaying === `${question.id}_melodic` ? 'audio-btn--playing' : ''
            }`}
            //onClick={() => handlePlayAudio('melodic')}
            disabled={loadingAudio === `${question.id}_melodic`}
          >
            {loadingAudio === `${question.id}_melodic` ? (
              <LoadingIndicator size="small" />
            ) : currentPlaying === `${question.id}_melodic` ? (
              <Square />
            ) : (
              <Play />
            )}
            <span>Поступово</span>
          </button>
        </div>

        {currentPlaying && (
          <button
            className="audio-btn audio-btn--stop"
            onClick={stopAudio}
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
        
        <div className="options-grid">
          {answerOptions.map((option) => (
            <button
              key={option.id}
              className={getOptionClass(option)}
              onClick={() => handleAnswerSelect(option.id)}
              disabled={isSubmitting || (hasAnswered && !isReviewMode)}
            >
              <div className="option-content">
                <span className="option-name">{option.name}</span>
                <span className="option-semitones">{option.semitones} півтонів</span>
              </div>
              {getOptionIcon(option)}
            </button>
          ))}
        </div>
      </div>

      {showCorrectAnswer && (
        <div className="interval-test__feedback">
          <div className={`feedback feedback--${selectedAnswer === question.interval_type ? 'correct' : 'wrong'}`}>
            {selectedAnswer === question.interval_type ? (
              <div className="feedback-content">
                <CheckCircle />
                <span>Правильно! Це дійсно {intervalOptions.find(opt => opt.id === question.interval_type)?.name}</span>
              </div>
            ) : (
              <div className="feedback-content">
                <XCircle />
                <span>
                  Неправильно. Правильна відповідь: {intervalOptions.find(opt => opt.id === question.interval_type)?.name}
                  {selectedAnswer && (
                    <span className="feedback-details">
                      <br />Ваша відповідь: {intervalOptions.find(opt => opt.id === selectedAnswer)?.name}
                    </span>
                  )}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default IntervalRecognitionTest;