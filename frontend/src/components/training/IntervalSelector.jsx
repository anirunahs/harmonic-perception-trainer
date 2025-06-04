import React from "react";

const IntervalSelector = ({
  selectedIntervals,
  onIntervalToggle,
  onSelectAll,
  onClear
}) => {
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

  return (
    <div className="setting-group">
      <div className="setting-group__header">
        <label className="setting-group__label">Інтервали для тренування</label>
        <div className="setting-group__actions">
          <button 
            className="btn btn--ghost btn--sm"
            onClick={onSelectAll}
          >
            Обрати всі
          </button>
          <button 
            className="btn btn--ghost btn--sm"
            onClick={onClear}
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
              onChange={() => onIntervalToggle(interval.id)}
            />
            <span className="interval-checkbox__checkmark"></span>
            <span className="interval-checkbox__label">
              {interval.name} ({interval.semitones} пт)
            </span>
          </label>
        ))}
      </div>
    </div>
  );
};

export default IntervalSelector;