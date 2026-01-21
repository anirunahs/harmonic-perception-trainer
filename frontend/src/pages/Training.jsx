import React, { useState, useEffect } from "react";
import Header from "../components/Header";
import TrainingSettings from "../components/training/TrainingSettings";
import GeneratedIntervals from "../components/training/GeneratedIntervals";
import ErrorMessage from "../components/training/ErrorMessage";
import PianoKeyboard from "../components/training/PianoKeyboard";
import { useInstrument } from "../hooks/useInstrument";

const Training = () => {
  const [selectedIntervals, setSelectedIntervals] = useState([]);
  const [selectedNote, setSelectedNote] = useState("C");
  const [selectedInstrument, setSelectedInstrument] = useState("piano");
  const [generatedIntervals, setGeneratedIntervals] = useState([]);
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
    console.log('Generated intervals:', intervalsData, 'Instrument:', selectedInstrument);
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
    setCurrentPlaying(null);
  };

  // Play interval using Tone.js (real samples)
  const handlePlayAudio = async (intervalId, playType) => {
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
      } else {
        await playMelodicInterval(baseNote, interval.interval_type, noteDuration);
      }
      
      // Calculate actual duration based on note length + release time
      // Half note at 120 BPM = 1 second, plus ~1.5s release for piano
      const baseDuration = 1000; // 1 second for half note
      const releaseTime = 1500; // 1.5 seconds for release
      const totalDuration = playType === 'harmonic' 
        ? baseDuration + releaseTime 
        : (baseDuration * 2) + releaseTime; // melodic plays two notes
      
      // Clear playing state after sound finishes
      setTimeout(() => {
        setCurrentPlaying(null);
      }, totalDuration);
      
    } catch (error) {
      console.error('Error playing interval:', error);
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
            <TrainingSettings
              selectedNote={selectedNote}
              setSelectedNote={setSelectedNote}
              selectedInstrument={selectedInstrument}
              setSelectedInstrument={setSelectedInstrument}
              selectedIntervals={selectedIntervals}
              onIntervalToggle={handleIntervalToggle}
              onSelectAllIntervals={handleSelectAllIntervals}
              onClearIntervals={handleClearIntervals}
              onGenerateIntervals={generateIntervals}
              onClearGenerated={clearGenerated}
              isGenerating={isLoading}
              isLoaded={isLoaded}
              hasGeneratedIntervals={generatedIntervals.length > 0}
            />

            <ErrorMessage 
              error={audioError} 
              onClose={() => setAudioError(null)} 
            />

            <GeneratedIntervals
              intervals={generatedIntervals}
              selectedNote={selectedNote}
              currentPlaying={currentPlaying}
              loadingAudio={isLoading}
              onPlayAudio={handlePlayAudio}
              onStopAudio={handleStopAudio}
              isGenerating={isLoading}
              instrumentLoaded={isLoaded}
            />
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