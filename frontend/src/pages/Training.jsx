import React, { useState, useRef, useEffect } from "react";
import { Play, Square, Volume2, VolumeX, Settings, Trash2, RotateCcw } from "lucide-react";
import Header from "../components/Header";
import LoadingIndicator from "../components/LoadingIndicator";
import api from "../api";

const Training = () => {
  const [selectedIntervals, setSelectedIntervals] = useState([]);
  const [selectedNote, setSelectedNote] = useState("C");
  const [generatedIntervals, setGeneratedIntervals] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentPlaying, setCurrentPlaying] = useState(null);
  const [audioError, setAudioError] = useState(null);
  const [loadingAudio, setLoadingAudio] = useState(null);
  const audioRefs = useRef({});

  const intervals = [
    { name: "Мала секунда", semitones: 1, id: "minor_second" },
    { name: "Велика секунда", semitones: 2, id: "major_second" },
    { name: "Мала терція", semitones: 3, id: "minor_third" },
    { name: "Велика терція", semitones: 4, id: "major_third" },
    { name: "Чиста кварта", semitones: 5, id: "perfect_fourth" },
    { name: "Тритон", semitones: 6, id: "tritone" },
    { name: "Чиста квінта", semitones: 7, id: "perfect_fifth" },
    { name: "Мала секста", semitones: 8, id: "minor_sixth" },
    { name: "Велика секста", semitones: 9, id: "major_sixth" },
    { name: "Мала септима", semitones: 10, id: "minor_seventh" },
    { name: "Велика септима", semitones: 11, id: "major_seventh" },
    { name: "Чиста октава", semitones: 12, id: "perfect_octave" }
  ];

  const notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

  useEffect(() => {
    return () => {
      Object.values(audioRefs.current).forEach(audio => {
        if (audio) {
          audio.pause();
          audio.src = '';
        }
      });
    };
  }, []);

  const handleIntervalToggle = (intervalId) => {
    setSelectedIntervals(prev => 
      prev.includes(intervalId)
        ? prev.filter(id => id !== intervalId)
        : [...prev, intervalId]
    );
  };

  const handleSelectAllIntervals = () => {
    setSelectedIntervals(intervals.map(interval => interval.id));
  };

  const handleClearIntervals = () => {
    setSelectedIntervals([]);
  };

  const generateIntervals = async () => {
    if (selectedIntervals.length === 0) {
      alert("Будь ласка, оберіть хоча б один інтервал");
      return;
    }

    setIsGenerating(true);
    setAudioError(null);
    
    try {
      const response = await api.post("/api/training/generate-intervals/", {
        intervals: selectedIntervals,
        base_note: selectedNote
      });

      setGeneratedIntervals(response.data.intervals);
      console.log('Generated intervals:', response.data.intervals);
    } catch (error) {
      console.error("Помилка генерації інтервалів:", error);
      setAudioError("Помилка при генерації інтервалів. Спробуйте пізніше.");
    } finally {
      setIsGenerating(false);
    }
  };

  const clearGenerated = () => {
    setCurrentPlaying(null);
    setAudioError(null);
    setLoadingAudio(null);

    Object.values(audioRefs.current).forEach(audio => {
      if (audio) {
        audio.onended = null;
        audio.onerror = null;
        audio.oncanplay = null;
        audio.onloadstart = null;

        audio.pause();
        audio.currentTime = 0;
        audio.src = '';
        audio.load();
      }
    });

    audioRefs.current = {};
    setGeneratedIntervals([]);
  };

  const playAudio = async (intervalId, playType) => {
    const playId = `${intervalId}_${playType}`;
    
    if (currentPlaying && audioRefs.current[currentPlaying]) {
      audioRefs.current[currentPlaying].pause();
      audioRefs.current[currentPlaying].currentTime = 0;
    }

    if (currentPlaying === playId) {
      setCurrentPlaying(null);
      return;
    }

    try {
      setAudioError(null);
      setLoadingAudio(playId);
      
      const interval = generatedIntervals.find(int => int.id === intervalId);
      if (!interval) {
        throw new Error("Інтервал не знайдено");
      }

      if (!audioRefs.current[playId]) {
        const audio = new Audio();
        audioRefs.current[playId] = audio;
        
        audio.preload = 'metadata';
        
        audio.onloadstart = () => {
          console.log(`Loading started for ${playId}`);
        };
        
        audio.oncanplay = () => {
          console.log(`Can play ${playId}`);
          setLoadingAudio(null);
        };
        
        audio.onended = () => {
          console.log(`Ended ${playId}`);
          setCurrentPlaying(null);
          setLoadingAudio(null);
        };
        
        audio.onerror = (e) => {
          if (!audioRefs.current[playId]) {
            return;
          }

          console.error("Audio error:", e);
          
          if (!audio.src || audio.src === '' || audio.src === window.location.href) {
            return;
          }
          
          let errorMessage = "Помилка відтворення аудіо";
          if (audio.error) {
            switch (audio.error.code) {
              case audio.error.MEDIA_ERR_ABORTED:
                errorMessage = "Відтворення було перервано";
                break;
              case audio.error.MEDIA_ERR_NETWORK:
                errorMessage = "Помилка мережі при завантаженні аудіо";
                break;
              case audio.error.MEDIA_ERR_DECODE:
                errorMessage = "Помилка декодування аудіо";
                break;
              case audio.error.MEDIA_ERR_SRC_NOT_SUPPORTED:
                errorMessage = "Формат аудіо не підтримується";
                break;
              default:
                errorMessage = `Невідома помилка аудіо (код: ${audio.error.code})`;
            }
          }
          
          setAudioError(errorMessage);
          setCurrentPlaying(null);
          setLoadingAudio(null);
        };
      }

      const audio = audioRefs.current[playId];
      
      const audioUrl = playType === 'harmonic' ? interval.harmonic_url : interval.melodic_url;
      
      try {
        const response = await api.get(audioUrl, {
          responseType: 'blob'
        });
        
        const audioBlob = new Blob([response.data], { type: 'audio/wav' });
        const blobUrl = URL.createObjectURL(audioBlob);
        
        audio.src = blobUrl;
        
        const originalOnended = audio.onended;
        audio.onended = () => {
          URL.revokeObjectURL(blobUrl);
          if (originalOnended) originalOnended();
        };
        
      } catch (error) {
        console.error("Помилка завантаження аудіо:", error);
        throw new Error("Не вдалося завантажити аудіофайл");
      }
      
      audio.load();
      setCurrentPlaying(playId);
      
      await new Promise((resolve, reject) => {
        const onCanPlay = () => {
          audio.removeEventListener('canplay', onCanPlay);
          audio.removeEventListener('error', onError);
          resolve();
        };
        
        const onError = (e) => {
          audio.removeEventListener('canplay', onCanPlay);
          audio.removeEventListener('error', onError);
          reject(e);
        };
        
        audio.addEventListener('canplay', onCanPlay);
        audio.addEventListener('error', onError);
        
        setTimeout(() => {
          audio.removeEventListener('canplay', onCanPlay);
          audio.removeEventListener('error', onError);
          reject(new Error('Таймаут завантаження аудіо'));
        }, 10000);
      });
      
      await audio.play();
      setLoadingAudio(null);
      
    } catch (error) {
      console.error("Помилка відтворення:", error);
      setAudioError(`Помилка відтворення: ${error.message}`);
      setCurrentPlaying(null);
      setLoadingAudio(null);
    }
  };

  const stopAudio = () => {
    if (currentPlaying && audioRefs.current[currentPlaying]) {
      const audio = audioRefs.current[currentPlaying];

      const originalOnError = audio.onerror;
      audio.onerror = null;
      audio.pause();
      audio.currentTime = 0;
      
      setTimeout(() => {
        if (audio) {
          audio.onerror = originalOnError;
        }
      }, 100);
      
      setCurrentPlaying(null);
    }
    setLoadingAudio(null);
  };

  const getIntervalName = (intervalId) => {
    return intervals.find(int => int.id === intervalId)?.name || intervalId;
  };

  return (
    <>
      <Header />
      <div className="training-page">
        <div className="training-page__container">
          <div className="training-page__header">
            <h1 className="training-page__title">Тренування музичних інтервалів</h1>
            <p className="training-page__subtitle">
              Оберіть інтервали та базову ноту для генерації аудіопрослуховування
            </p>
          </div>

          <div className="training-page__content">
            <div className="training-settings">
              <div className="training-settings__header">
                <h2 className="training-settings__title">
                  <Settings />
                  Налаштування тренування
                </h2>
              </div>

              <div className="training-settings__body">
                <div className="setting-group">
                  <label className="setting-group__label">Базова нота</label>
                  <div className="note-selector">
                    {notes.map(note => (
                      <button
                        key={note}
                        className={`note-selector__button ${selectedNote === note ? 'note-selector__button--active' : ''}`}
                        onClick={() => setSelectedNote(note)}
                      >
                        {note}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="setting-group">
                  <div className="setting-group__header">
                    <label className="setting-group__label">Інтервали для тренування</label>
                    <div className="setting-group__actions">
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
                  
                  <div className="interval-selector">
                    {intervals.map(interval => (
                      <label key={interval.id} className="interval-checkbox">
                        <input
                          type="checkbox"
                          checked={selectedIntervals.includes(interval.id)}
                          onChange={() => handleIntervalToggle(interval.id)}
                        />
                        <span className="interval-checkbox__checkmark"></span>
                        <span className="interval-checkbox__label">
                          {interval.name} ({interval.semitones} пт)
                        </span>
                      </label>
                    ))}
                  </div>
                </div>

                <div className="setting-group">
                  <div className="setting-actions">
                    <button
                      className="btn btn--primary"
                      onClick={generateIntervals}
                      disabled={isGenerating || selectedIntervals.length === 0}
                    >
                      {isGenerating ? (
                        <>
                          <LoadingIndicator size="small" />
                          <span>Генерація...</span>
                        </>
                      ) : (
                        <>
                          <Volume2 />
                          <span>Згенерувати інтервали</span>
                        </>
                      )}
                    </button>

                    {generatedIntervals.length > 0 && (
                      <button
                        className="btn btn--ghost"
                        onClick={clearGenerated}
                      >
                        <Trash2 />
                        <span>Очистити</span>
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {audioError && (
              <div className="status-card status-card--error">
                <div className="status-card__icon">
                  <VolumeX />
                </div>
                <div className="status-card__content">
                  <h3 className="status-card__title">Помилка відтворення</h3>
                  <p className="status-card__description">{audioError}</p>
                  <button 
                    className="btn btn--ghost btn--sm"
                    onClick={() => setAudioError(null)}
                  >
                    Закрити
                  </button>
                </div>
              </div>
            )}

            {generatedIntervals.length > 0 && (
              <div className="generated-intervals">
                <div className="generated-intervals__header">
                  <h2 className="generated-intervals__title">
                    Згенеровані інтервали
                  </h2>
                  <div className="generated-intervals__info">
                    Базова нота: <strong>{selectedNote}</strong> | 
                    Кількість інтервалів: <strong>{generatedIntervals.length}</strong>
                  </div>
                </div>

                <div className="intervals-grid">
                  {generatedIntervals.map(interval => (
                    <div key={interval.id} className="interval-card">
                      <div className="interval-card__header">
                        <h3 className="interval-card__title">
                          {getIntervalName(interval.interval_type)}
                        </h3>
                        <div className="interval-card__notes">
                          {interval.base_note} → {interval.target_note}
                        </div>
                      </div>

                      <div className="interval-card__controls">
                        <button
                          className={`interval-play-btn ${
                            currentPlaying === `${interval.id}_harmonic` ? 'interval-play-btn--playing' : ''
                          } ${
                            loadingAudio === `${interval.id}_harmonic` ? 'interval-play-btn--loading' : ''
                          }`}
                          onClick={() => playAudio(interval.id, 'harmonic')}
                          disabled={isGenerating || loadingAudio === `${interval.id}_harmonic`}
                        >
                          {loadingAudio === `${interval.id}_harmonic` ? (
                            <LoadingIndicator size="small" />
                          ) : currentPlaying === `${interval.id}_harmonic` ? (
                            <Square />
                          ) : (
                            <Play />
                          )}
                          <span>Цілісно</span>
                        </button>

                        <button
                          className={`interval-play-btn interval-play-btn--secondary ${
                            currentPlaying === `${interval.id}_melodic` ? 'interval-play-btn--playing' : ''
                          } ${
                            loadingAudio === `${interval.id}_melodic` ? 'interval-play-btn--loading' : ''
                          }`}
                          onClick={() => playAudio(interval.id, 'melodic')}
                          disabled={isGenerating || loadingAudio === `${interval.id}_melodic`}
                        >
                          {loadingAudio === `${interval.id}_melodic` ? (
                            <LoadingIndicator size="small" />
                          ) : currentPlaying === `${interval.id}_melodic` ? (
                            <Square />
                          ) : (
                            <Play />
                          )}
                          <span>Поступово</span>
                        </button>
                      </div>

                      {currentPlaying && currentPlaying.startsWith(interval.id) && (
                        <div className="interval-card__status">
                          <div className="playing-indicator">
                            <div className="playing-indicator__bars">
                              <div className="playing-indicator__bar"></div>
                              <div className="playing-indicator__bar"></div>
                              <div className="playing-indicator__bar"></div>
                            </div>
                            <span>Відтворюється...</span>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {currentPlaying && (
                  <div className="global-controls">
                    <button
                      className="btn btn--danger btn--sm"
                      onClick={stopAudio}
                    >
                      <Square />
                      <span>Зупинити відтворення</span>
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
};

export default Training;