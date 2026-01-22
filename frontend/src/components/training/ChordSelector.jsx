import React from "react";
import { CheckSquare, Square } from "lucide-react";

const CHORDS = [
  { name: "Мажорний", id: "major" },
  { name: "Мінорний", id: "minor" },
  { name: "Зменшений", id: "diminished" },
  { name: "Збільшений", id: "augmented" },
  { name: "Мажорний септакорд", id: "major_seventh" },
  { name: "Мінорний септакорд", id: "minor_seventh" },
  { name: "Домінантний септакорд", id: "dominant_seventh" },
  { name: "Квартовий затриманий", id: "suspended_fourth" },
  { name: "Секундовий затриманий", id: "suspended_second" },
];

const ChordSelector = ({
  selectedChords,
  onChordToggle,
  onSelectAll,
  onClear,
}) => {
  return (
    <div className="setting-group">
      <label className="setting-group__label">
        Оберіть акорди
      </label>
      <div className="chord-selector">
        <div className="chord-selector__actions">
          <button
            type="button"
            className="btn btn--link btn--sm"
            onClick={onSelectAll}
          >
            Обрати всі
          </button>
          <button
            type="button"
            className="btn btn--link btn--sm"
            onClick={onClear}
          >
            Очистити
          </button>
        </div>
        <div className="chord-selector__list">
          {CHORDS.map((chord) => (
            <button
              key={chord.id}
              type="button"
              className={`chord-item ${
                selectedChords.includes(chord.id) ? "chord-item--selected" : ""
              }`}
              onClick={() => onChordToggle(chord.id)}
            >
              {selectedChords.includes(chord.id) ? (
                <CheckSquare size={18} />
              ) : (
                <Square size={18} />
              )}
              <span>{chord.name}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ChordSelector;
