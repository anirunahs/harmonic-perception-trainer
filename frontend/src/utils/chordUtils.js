/**
 * Utility functions for chord calculations
 */

const CHROMATIC_SCALE = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

const CHORDS_SEMITONES = {
  'major': [0, 4, 7],
  'minor': [0, 3, 7],
  'diminished': [0, 3, 6],
  'augmented': [0, 4, 8],
  'major_seventh': [0, 4, 7, 11],
  'minor_seventh': [0, 3, 7, 10],
  'dominant_seventh': [0, 4, 7, 10],
  'suspended_fourth': [0, 5, 7],
  'suspended_second': [0, 2, 7],
};

/**
 * Get target note from base note and semitones
 */
export const getTargetNote = (baseNote, semitones) => {
  // Handle enharmonic equivalents
  const noteMap = {
    'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'
  };
  const normalizedNote = noteMap[baseNote] || baseNote;
  
  const baseIndex = CHROMATIC_SCALE.indexOf(normalizedNote);
  if (baseIndex === -1) {
    return baseNote;
  }
  
  const targetIndex = (baseIndex + semitones) % 12;
  return CHROMATIC_SCALE[targetIndex];
};

/**
 * Calculate all notes in a chord
 */
export const getChordNotes = (rootNote, chordType, octave = 4) => {
  const semitonesList = CHORDS_SEMITONES[chordType];
  if (!semitonesList) {
    return [];
  }
  
  const notes = semitonesList.map(semitones => {
    if (semitones === 0) {
      return { note: rootNote, octave, semitones: 0 };
    }
    
    const targetNote = getTargetNote(rootNote, semitones);
    const noteOctave = octave + Math.floor(semitones / 12);
    
    return {
      note: targetNote,
      octave: noteOctave,
      semitones,
      noteWithOctave: `${targetNote}${noteOctave}`
    };
  });
  
  return notes;
};

/**
 * Get chord structure description
 */
export const getChordStructure = (chordType) => {
  const structures = {
    'major': '1-3-5 (Мажорна терція + Квінта)',
    'minor': '1-♭3-5 (Мінорна терція + Квінта)',
    'diminished': '1-♭3-♭5 (Мінорна терція + Зменшена квінта)',
    'augmented': '1-3-♯5 (Мажорна терція + Збільшена квінта)',
    'major_seventh': '1-3-5-7 (Мажорний + Велика септима)',
    'minor_seventh': '1-♭3-5-♭7 (Мінорний + Мала септима)',
    'dominant_seventh': '1-3-5-♭7 (Мажорний + Мала септима)',
    'suspended_fourth': '1-4-5 (Кварта замість терції)',
    'suspended_second': '1-2-5 (Секунда замість терції)',
  };
  return structures[chordType] || chordType;
};

/**
 * Get chord sound description
 */
export const getChordSound = (chordType) => {
  const sounds = {
    'major': 'Світлий, радісний',
    'minor': 'Сумний, м\'який',
    'diminished': 'Напружений, нестабільний',
    'augmented': 'Екзотичний, нестабільний',
    'major_seventh': 'Багатий, джазовий',
    'minor_seventh': 'М\'який, джазовий',
    'dominant_seventh': 'Напружений, потребує розв\'язки',
    'suspended_fourth': 'Відкритий, невизначений',
    'suspended_second': 'Відкритий, невизначений',
  };
  return sounds[chordType] || '';
};
