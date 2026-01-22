import React from "react";
import { Music2, Music4 } from "lucide-react";

const MUSIC_ELEMENT_TYPES = [
  { id: "intervals", name: "Інтервали", icon: Music2 },
  { id: "chords", name: "Акорди", icon: Music4 },
];

const MusicElementTypeSelector = ({ selectedType, onTypeSelect }) => {
  return (
    <div className="setting-group">
      <label className="setting-group__label">
        <Music2 size={18} />
        Тип музичних елементів
      </label>
      <div className="music-type-selector">
        {MUSIC_ELEMENT_TYPES.map((type) => {
          const Icon = type.icon;
          return (
            <button
              key={type.id}
              className={`music-type-btn ${
                selectedType === type.id ? "music-type-btn--active" : ""
              }`}
              onClick={() => onTypeSelect(type.id)}
              title={type.name}
            >
              <Icon className="music-type-btn__icon" size={24} />
              <span className="music-type-btn__name">{type.name}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default MusicElementTypeSelector;
