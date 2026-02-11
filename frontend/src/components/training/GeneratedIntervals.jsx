import React from "react";
import { Square } from "lucide-react";
import IntervalCard from "./IntervalCard";

const GeneratedIntervals = ({
  intervals,
  selectedNote,
  currentPlaying,
  loadingAudio,
  onPlayAudio,
  onStopAudio,
  isGenerating
}) => {
  const getIntervalName = (intervalId) => {
    const intervalsList = [
      { name: "Мала секунда", id: "minor_second" },
      { name: "Велика секунда", id: "major_second" },
      { name: "Мала терція", id: "minor_third" },
      { name: "Велика терція", id: "major_third" },
      { name: "Чиста кварта", id: "perfect_fourth" },
      { name: "Тритон", id: "tritone" },
      { name: "Чиста квінта", id: "perfect_fifth" },
      { name: "Мала секста", id: "minor_sixth" },
      { name: "Велика секста", id: "major_sixth" },
      { name: "Мала септима", id: "minor_seventh" },
      { name: "Велика септима", id: "major_seventh" },
      { name: "Чиста октава", id: "perfect_octave" }
    ];
    return intervalsList.find(int => int.id === intervalId)?.name || intervalId;
  };

  if (intervals.length === 0) {
    return null;
  }

  return (
    <div className="generated-intervals">
      <div className="generated-intervals__header">
        <h2 className="generated-intervals__title">
          Згенеровані інтервали
        </h2>
        <div className="generated-intervals__info">
          Базова нота: <strong>{selectedNote}</strong> | 
          Кількість інтервалів: <strong>{intervals.length}</strong>
        </div>
      </div>

      <div className="intervals-grid">
        {intervals.map(interval => (
          <IntervalCard
            key={interval.id}
            interval={interval}
            intervalName={getIntervalName(interval.interval_type)}
            currentPlaying={currentPlaying}
            loadingAudio={loadingAudio}
            onPlayAudio={onPlayAudio}
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

export default GeneratedIntervals;