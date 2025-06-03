import React, { useState, useRef } from "react";
import { Play, Square, Volume2, VolumeX, Settings, Trash2, RotateCcw } from "lucide-react";
import Header from "../components/Header";
import api from "../api";

const Training = () => {
  const [selectedIntervals, setSelectedIntervals] = useState([]);
  const [selectedNote, setSelectedNote] = useState("C");
  const [generatedIntervals, setGeneratedIntervals] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentPlaying, setCurrentPlaying] = useState(null);
  const [lastPlayedNote, setLastPlayedNote] = useState(null);
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
                    >
                      <Volume2 />
                      <span>Згенерувати інтервали</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );


};

export default Training