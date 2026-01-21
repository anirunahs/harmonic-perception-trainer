import React from "react";
import { Play, Square } from "lucide-react";
import LoadingIndicator from "../LoadingIndicator";
import PlayingIndicator from "./PlayingIndicator";

const IntervalCard = ({
  interval,
  intervalName,
  currentPlaying,
  loadingAudio,
  onPlayAudio,
  isGenerating
}) => {
  // currentPlaying can be object { id, type } or string (for backwards compatibility)
  const currentId = typeof currentPlaying === 'object' ? currentPlaying?.id : currentPlaying;
  const currentType = typeof currentPlaying === 'object' ? currentPlaying?.type : null;
  
  const isHarmonicPlaying = currentId === interval.id && currentType === 'harmonic';
  const isMelodicPlaying = currentId === interval.id && currentType === 'melodic';
  const isHarmonicLoading = loadingAudio === `${interval.id}_harmonic`;
  const isMelodicLoading = loadingAudio === `${interval.id}_melodic`;
  const isCurrentIntervalPlaying = currentId === interval.id;

  return (
    <div className="interval-card">
      <div className="interval-card__header">
        <h3 className="interval-card__title">
          {intervalName}
        </h3>
        <div className="interval-card__notes">
          {interval.base_note} → {interval.target_note}
        </div>
      </div>

      <div className="interval-card__controls">
        <button
          className={`interval-play-btn ${isHarmonicPlaying ? 'interval-play-btn--playing' : ''} ${isHarmonicLoading ? 'interval-play-btn--loading' : ''}`}
          onClick={() => onPlayAudio(interval.id, 'harmonic')}
          disabled={isGenerating || isHarmonicLoading}
        >
          {isHarmonicLoading ? (
            <LoadingIndicator size="small" />
          ) : isHarmonicPlaying ? (
            <Square />
          ) : (
            <Play />
          )}
          <span>Цілісно</span>
        </button>

        <button
          className={`interval-play-btn interval-play-btn--secondary ${isMelodicPlaying ? 'interval-play-btn--playing' : ''} ${isMelodicLoading ? 'interval-play-btn--loading' : ''}`}
          onClick={() => onPlayAudio(interval.id, 'melodic')}
          disabled={isGenerating || isMelodicLoading}
        >
          {isMelodicLoading ? (
            <LoadingIndicator size="small" />
          ) : isMelodicPlaying ? (
            <Square />
          ) : (
            <Play />
          )}
          <span>Поступово</span>
        </button>
      </div>

      {isCurrentIntervalPlaying && (
        <div className="interval-card__status">
          <PlayingIndicator />
        </div>
      )}
    </div>
  );
};

export default IntervalCard;