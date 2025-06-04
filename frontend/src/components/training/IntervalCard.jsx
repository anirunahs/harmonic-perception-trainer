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
  const isHarmonicPlaying = currentPlaying === `${interval.id}_harmonic`;
  const isMelodicPlaying = currentPlaying === `${interval.id}_melodic`;
  const isHarmonicLoading = loadingAudio === `${interval.id}_harmonic`;
  const isMelodicLoading = loadingAudio === `${interval.id}_melodic`;
  const isCurrentIntervalPlaying = currentPlaying && currentPlaying.startsWith(interval.id);

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