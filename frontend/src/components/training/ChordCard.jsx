import React from "react";
import { Play, Square } from "lucide-react";
import { getChordNotes } from "../../utils/chordUtils";

const ChordCard = ({
  chord,
  chordName,
  currentPlaying,
  loadingAudio,
  onPlayAudio,
  isGenerating,
}) => {
  const isHarmonicPlaying = currentPlaying?.id === chord.id && currentPlaying?.type === 'harmonic';
  const isMelodicPlaying = currentPlaying?.id === chord.id && currentPlaying?.type === 'melodic';
  const isCurrentChordPlaying = currentPlaying?.id === chord.id;

  // Calculate chord notes
  const chordNotes = getChordNotes(chord.root_note, chord.chord_type, 4);
  const notesDisplay = chordNotes.map(n => n.note).join(' - ');

  return (
    <div className={`interval-card ${isCurrentChordPlaying ? "interval-card--playing" : ""}`}>
      <div className="interval-card__header">
        <h3 className="interval-card__title">
          {chordName}
        </h3>
        <div className="interval-card__notes">
          {chord.root_note} → {notesDisplay}
        </div>
      </div>

      <div className="interval-card__controls">
        <button
          className={`interval-play-btn ${
            isHarmonicPlaying ? 'interval-play-btn--playing' : ''
          }`}
          onClick={() => onPlayAudio(chord.id, 'harmonic')}
          disabled={isGenerating || loadingAudio}
        >
          {isHarmonicPlaying ? (
            <Square />
          ) : (
            <Play />
          )}
          <span>Одночасно</span>
        </button>

        <button
          className={`interval-play-btn interval-play-btn--secondary ${
            isMelodicPlaying ? 'interval-play-btn--playing' : ''
          }`}
          onClick={() => onPlayAudio(chord.id, 'melodic')}
          disabled={isGenerating || loadingAudio}
        >
          {isMelodicPlaying ? (
            <Square />
          ) : (
            <Play />
          )}
          <span>Послідовно</span>
        </button>
      </div>

      {isCurrentChordPlaying && (
        <div className="interval-card__status">
          <div className="playing-indicator">
            <div className="playing-indicator__bars">
              <div className="playing-indicator__bar"></div>
              <div className="playing-indicator__bar"></div>
              <div className="playing-indicator__bar"></div>
            </div>
            <span>Відтворення...</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChordCard;
