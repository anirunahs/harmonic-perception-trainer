import React from "react";
import { Square } from "lucide-react";
import ChordCard from "./ChordCard";

const GeneratedChords = ({
  chords,
  selectedNote,
  currentPlaying,
  loadingAudio,
  onPlayAudio,
  onStopAudio,
  isGenerating
}) => {
  const getChordName = (chordId) => {
    const chordsList = [
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
    return chordsList.find(c => c.id === chordId)?.name || chordId;
  };

  if (chords.length === 0) {
    return null;
  }

  return (
    <div className="generated-intervals">
      <div className="generated-intervals__header">
        <h2 className="generated-intervals__title">
          Згенеровані акорди
        </h2>
        <div className="generated-intervals__info">
          Базова нота: <strong>{selectedNote}</strong> | 
          Кількість акордів: <strong>{chords.length}</strong>
        </div>
      </div>

      <div className="intervals-grid">
        {chords.map(chord => (
          <ChordCard
            key={chord.id}
            chord={chord}
            chordName={getChordName(chord.chord_type)}
            currentPlaying={currentPlaying}
            loadingAudio={loadingAudio}
            onPlayAudio={(chordId, playType) => onPlayAudio(chordId, playType)}
            isGenerating={isGenerating}
          />
        ))}
      </div>

      {currentPlaying && (
        <div className="global-controls">
          <button
            className="btn btn--stop-playback btn--sm"
            onClick={onStopAudio}
          >
            <Square />
            <span>Зупинити відтворення</span>
          </button>
        </div>
      )}
    </div>
  );
};

export default GeneratedChords;
