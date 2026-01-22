import React, { useState, useEffect } from "react";
import * as Tone from "tone";
import Header from "../components/Header";
import TrainingSettings from "../components/training/TrainingSettings";
import GeneratedIntervals from "../components/training/GeneratedIntervals";
import GeneratedChords from "../components/training/GeneratedChords";
import ErrorMessage from "../components/training/ErrorMessage";
import PianoKeyboard from "../components/training/PianoKeyboard";
import { useInstrument } from "../hooks/useInstrument";
import { getChordNotes } from "../utils/chordUtils";

const Training = () => {
  const [musicElementType, setMusicElementType] = useState("intervals"); // "intervals" or "chords"
  const [selectedIntervals, setSelectedIntervals] = useState([]);
  const [selectedChords, setSelectedChords] = useState([]);
  const [selectedNote, setSelectedNote] = useState("C");
  const [selectedInstrument, setSelectedInstrument] = useState("piano");
  const [generatedIntervals, setGeneratedIntervals] = useState([]);
  const [generatedChords, setGeneratedChords] = useState([]);
  const [currentPlaying, setCurrentPlaying] = useState(null);
  const [audioError, setAudioError] = useState(null);

  // Use Tone.js instrument with real samples
  const {
    isLoaded,
    isLoading,
    error: instrumentError,
    playNote,
    playHarmonicInterval,
    playMelodicInterval,
    playChord,
    playMelodicChord,
    stopAll,
    changeInstrument,
  } = useInstrument(selectedInstrument);

  // Change instrument when selection changes
  useEffect(() => {
    changeInstrument(selectedInstrument);
  }, [selectedInstrument, changeInstrument]);

  // Show instrument loading errors
  useEffect(() => {
    if (instrumentError) {
      setAudioError(instrumentError);
    }
  }, [instrumentError]);

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

  const chords = [
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

  const handleChordToggle = (chordId) => {
    setSelectedChords(prev => 
      prev.includes(chordId)
        ? prev.filter(id => id !== chordId)
        : [...prev, chordId]
    );
  };

  const handleSelectAllChords = () => {
    setSelectedChords(chords.map(chord => chord.id));
  };

  const handleClearChords = () => {
    setSelectedChords([]);
  };

  // Generate intervals list (no API call - sound is generated client-side)
  const generateIntervals = () => {
    if (selectedIntervals.length === 0) {
      alert("Будь ласка, оберіть хоча б один інтервал");
      return;
    }

    if (!isLoaded) {
      setAudioError("Інструмент ще завантажується. Зачекайте...");
      return;
    }

    setAudioError(null);
    
    // Create interval objects for display
    const intervalsData = selectedIntervals.map((intervalId, index) => {
      const intervalInfo = intervals.find(i => i.id === intervalId);
      const baseNoteWithOctave = `${selectedNote}4`;
      
      return {
        id: `${intervalId}-${index}`,
        interval_type: intervalId,
        base_note: selectedNote,
        target_note: getTargetNote(selectedNote, intervalInfo.semitones),
        semitones: intervalInfo.semitones,
        baseNoteWithOctave,
      };
    });

    setGeneratedIntervals(intervalsData);
    setGeneratedChords([]);
    console.log('Generated intervals:', intervalsData, 'Instrument:', selectedInstrument);
  };

  // Generate chords list
  const generateChords = () => {
    if (selectedChords.length === 0) {
      alert("Будь ласка, оберіть хоча б один акорд");
      return;
    }

    if (!isLoaded) {
      setAudioError("Інструмент ще завантажується. Зачекайте...");
      return;
    }

    setAudioError(null);
    
    // Create chord objects for display
    const chordsData = selectedChords.map((chordId, index) => {
      const rootNoteWithOctave = `${selectedNote}4`;
      
      return {
        id: `${chordId}-${index}`,
        chord_type: chordId,
        root_note: selectedNote,
        rootNoteWithOctave: rootNoteWithOctave,
      };
    });

    setGeneratedChords(chordsData);
    setGeneratedIntervals([]);
    console.log('Generated chords:', chordsData, 'Instrument:', selectedInstrument);
  };

  // Calculate target note from base note and semitones
  const getTargetNote = (baseNote, semitones) => {
    const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const baseIndex = notes.indexOf(baseNote);
    const targetIndex = (baseIndex + semitones) % 12;
    return notes[targetIndex];
  };

  const clearGenerated = () => {
    stopAll();
    setGeneratedIntervals([]);
    setGeneratedChords([]);
    setCurrentPlaying(null);
  };

  // Play interval using Tone.js (real samples)
  const handlePlayInterval = async (intervalId, playType) => {
    const interval = generatedIntervals.find(i => i.id === intervalId);
    if (!interval) return;

    // Stop any currently playing sound
    stopAll();
    setCurrentPlaying({ id: intervalId, type: playType });
    
    try {
      const baseNote = `${interval.base_note}4`; // Use octave 4
      const noteDuration = '2n'; // Half note
      
      if (playType === 'harmonic') {
        await playHarmonicInterval(baseNote, interval.interval_type, noteDuration);
        
        // Calculate actual duration: note duration + release time + buffer
        const noteDurationSeconds = Tone.Time(noteDuration).toSeconds();
        // Use release time from instrument config (piano: 1.5s, guitar: 1.2s)
        const releaseTime = selectedInstrument === 'piano' ? 1.5 : 1.2;
        const buffer = 0.5; // Extra buffer to ensure sound finishes completely
        const totalDuration = (noteDurationSeconds + releaseTime + buffer) * 1000;
        
        setTimeout(() => {
          setCurrentPlaying(null);
        }, totalDuration);
      } else {
        await playMelodicInterval(baseNote, interval.interval_type, noteDuration);
        
        // Melodic: two notes sequentially, each with release
        const noteDurationSeconds = Tone.Time(noteDuration).toSeconds();
        const releaseTime = selectedInstrument === 'piano' ? 1.5 : 1.2;
        const buffer = 0.5;
        // First note duration + gap + second note duration + release + buffer
        const totalDuration = (noteDurationSeconds * 2 + releaseTime + buffer) * 1000;
        
        setTimeout(() => {
          setCurrentPlaying(null);
        }, totalDuration);
      }
      
    } catch (error) {
      console.error('Error playing interval:', error);
      setAudioError('Помилка відтворення звуку');
      setCurrentPlaying(null);
    }
  };

  // Play chord using Tone.js
  const handlePlayChord = async (chordId, playType) => {
    const chord = generatedChords.find(c => c.id === chordId);
    if (!chord) return;

    // Stop any currently playing sound
    stopAll();
    setCurrentPlaying({ id: chordId, type: playType });
    
    try {
      const rootNote = `${chord.root_note}4`; // Use octave 4
      const noteDuration = '2n'; // Half note
      
      if (playType === 'harmonic') {
        await playChord(rootNote, chord.chord_type, noteDuration);
        
        // Calculate actual duration: note duration + release time + buffer
        const noteDurationSeconds = Tone.Time(noteDuration).toSeconds();
        const releaseTime = selectedInstrument === 'piano' ? 1.5 : 1.2;
        const buffer = 0.7; // Extra buffer for chord (more notes = more time)
        const totalDuration = (noteDurationSeconds + releaseTime + buffer) * 1000;
        
        setTimeout(() => {
          setCurrentPlaying(null);
        }, totalDuration);
      } else {
        // Melodic: play notes sequentially
        await playMelodicChord(rootNote, chord.chord_type, noteDuration, 0.1);
        
        // Calculate duration for all notes
        const noteDurationSeconds = Tone.Time(noteDuration).toSeconds();
        const releaseTime = selectedInstrument === 'piano' ? 1.5 : 1.2;
        const gap = 0.1; // Gap between notes in seconds
        const chordNotes = getChordNotes(chord.root_note, chord.chord_type, 4);
        const notesCount = chordNotes.length;
        // Each note plays for noteDuration, with gaps between, plus final release
        const totalDuration = (
          (noteDurationSeconds * notesCount) + 
          (gap * (notesCount - 1)) + 
          releaseTime + 
          0.7 // buffer
        ) * 1000;
        
        setTimeout(() => {
          setCurrentPlaying(null);
        }, totalDuration);
      }
      
    } catch (error) {
      console.error('Error playing chord:', error);
      setAudioError('Помилка відтворення звуку');
      setCurrentPlaying(null);
    }
  };

  const handleStopAudio = () => {
    stopAll();
    setCurrentPlaying(null);
  };

  const handlePianoNotePlay = async (note, octave, frequency) => {
    if (!isLoaded) return;
    const noteWithOctave = `${note}${octave}`;
    console.log(`Playing: ${noteWithOctave} (${frequency.toFixed(2)}Hz)`);
    await playNote(noteWithOctave, '4n');
  };

  const handleGenerate = () => {
    if (musicElementType === "intervals") {
      generateIntervals();
    } else {
      generateChords();
    }
  };

  const hasGenerated = musicElementType === "intervals" 
    ? generatedIntervals.length > 0 
    : generatedChords.length > 0;

  const canGenerate = musicElementType === "intervals"
    ? selectedIntervals.length > 0
    : selectedChords.length > 0;

  return (
    <>
      <Header />
      <div className="training-page">
        <div className="training-page__container">
          <div className="training-page__header">
            <h1 className="training-page__title">Тренування музичних елементів</h1>
            <p className="training-page__subtitle">
              Оберіть інтервали або акорди та базову ноту для генерації аудіопрослуховування
            </p>
          </div>

          <div className="training-page__content">
            <TrainingSettings
              musicElementType={musicElementType}
              onMusicElementTypeChange={setMusicElementType}
              selectedNote={selectedNote}
              setSelectedNote={setSelectedNote}
              selectedInstrument={selectedInstrument}
              setSelectedInstrument={setSelectedInstrument}
              selectedIntervals={selectedIntervals}
              onIntervalToggle={handleIntervalToggle}
              onSelectAllIntervals={handleSelectAllIntervals}
              onClearIntervals={handleClearIntervals}
              selectedChords={selectedChords}
              onChordToggle={handleChordToggle}
              onSelectAllChords={handleSelectAllChords}
              onClearChords={handleClearChords}
              onGenerate={handleGenerate}
              onClearGenerated={clearGenerated}
              isGenerating={isLoading}
              isLoaded={isLoaded}
              hasGenerated={hasGenerated}
              canGenerate={canGenerate}
            />

            <ErrorMessage 
              error={audioError} 
              onClose={() => setAudioError(null)} 
            />

            {musicElementType === "intervals" ? (
              <GeneratedIntervals
                intervals={generatedIntervals}
                selectedNote={selectedNote}
                currentPlaying={currentPlaying}
                loadingAudio={isLoading}
                onPlayAudio={handlePlayInterval}
                onStopAudio={handleStopAudio}
                isGenerating={isLoading}
                instrumentLoaded={isLoaded}
              />
            ) : (
              <GeneratedChords
                chords={generatedChords}
                selectedNote={selectedNote}
                currentPlaying={currentPlaying}
                loadingAudio={isLoading}
                onPlayAudio={handlePlayChord}
                onStopAudio={handleStopAudio}
                isGenerating={isLoading}
                instrumentLoaded={isLoaded}
              />
            )}
          </div>
        </div>
      </div>
      
      <PianoKeyboard 
        isFixed={true}
        showNoteNames={true}
        showOctaves={true}
        startOctave={2}
        endOctave={4}
        onNotePlay={handlePianoNotePlay}
      />
    </>
  );
};

export default Training;
