import React, { useState } from "react";
import Header from "../components/Header";
import TrainingSettings from "../components/training/TrainingSettings";
import GeneratedIntervals from "../components/training/GeneratedIntervals";
import ErrorMessage from "../components/training/ErrorMessage";
import { useAudioPlayer } from "../hooks/useAudioPlayer";
import api from "../api";

const Training = () => {
  const [selectedIntervals, setSelectedIntervals] = useState([]);
  const [selectedNote, setSelectedNote] = useState("C");
  const [generatedIntervals, setGeneratedIntervals] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);

  const {
    currentPlaying,
    loadingAudio,
    audioError,
    setAudioError,
    playAudio,
    stopAudio,
    clearAllAudio
  } = useAudioPlayer();

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

  const generateIntervals = async () => {
    if (selectedIntervals.length === 0) {
      alert("Будь ласка, оберіть хоча б один інтервал");
      return;
    }

    setIsGenerating(true);
    setAudioError(null);
    
    try {
      const response = await api.post("/api/training/generate-intervals/", {
        intervals: selectedIntervals,
        base_note: selectedNote
      });

      setGeneratedIntervals(response.data.intervals);
      console.log('Generated intervals:', response.data.intervals);
    } catch (error) {
      console.error("Помилка генерації інтервалів:", error);
      setAudioError("Помилка при генерації інтервалів. Спробуйте пізніше.");
    } finally {
      setIsGenerating(false);
    }
  };

  const clearGenerated = () => {
    clearAllAudio();
    setGeneratedIntervals([]);
  };

  const handlePlayAudio = (intervalId, playType) => {
    playAudio(intervalId, playType, generatedIntervals);
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
              selectedIntervals={selectedIntervals}
              onIntervalToggle={handleIntervalToggle}
              onSelectAllIntervals={handleSelectAllIntervals}
              onClearIntervals={handleClearIntervals}
              onGenerateIntervals={generateIntervals}
              onClearGenerated={clearGenerated}
              isGenerating={isGenerating}
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
              loadingAudio={loadingAudio}
              onPlayAudio={handlePlayAudio}
              onStopAudio={stopAudio}
              isGenerating={isGenerating}
            />
          </div>
        </div>
      </div>
    </>
  );
};

export default Training;