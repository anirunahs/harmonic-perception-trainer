import React from "react";

const NoteSelector = ({ selectedNote, onNoteSelect }) => {
  const notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

  return (
    <div className="setting-group">
      <label className="setting-group__label">Базова нота</label>
      <div className="note-selector">
        {notes.map(note => (
          <button
            key={note}
            className={`note-selector__button ${selectedNote === note ? 'note-selector__button--active' : ''}`}
            onClick={() => onNoteSelect(note)}
          >
            {note}
          </button>
        ))}
      </div>
    </div>
  );
};

export default NoteSelector;