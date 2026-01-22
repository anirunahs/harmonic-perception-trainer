import React from "react";
import { Settings, Volume2, Trash2 } from "lucide-react";
import LoadingIndicator from "../LoadingIndicator";
import NoteSelector from "./NoteSelector";
import IntervalSelector from "./IntervalSelector";
import ChordSelector from "./ChordSelector";
import InstrumentSelector from "./InstrumentSelector";
import MusicElementTypeSelector from "./MusicElementTypeSelector";

const TrainingSettings = ({
  musicElementType,
  onMusicElementTypeChange,
  selectedNote,
  setSelectedNote,
  selectedInstrument,
  setSelectedInstrument,
  selectedIntervals,
  onIntervalToggle,
  onSelectAllIntervals,
  onClearIntervals,
  selectedChords,
  onChordToggle,
  onSelectAllChords,
  onClearChords,
  onGenerate,
  onClearGenerated,
  isGenerating,
  hasGenerated,
  canGenerate
}) => {
  return (
    <div className="training-settings">
      <div className="training-settings__header">
        <h2 className="training-settings__title">
          <Settings />
          Налаштування тренування
        </h2>
      </div>

      <div className="training-settings__body">
        <MusicElementTypeSelector
          selectedType={musicElementType}
          onTypeSelect={onMusicElementTypeChange}
        />

        <NoteSelector
          selectedNote={selectedNote}
          onNoteSelect={setSelectedNote}
        />

        <InstrumentSelector
          selectedInstrument={selectedInstrument}
          onInstrumentSelect={setSelectedInstrument}
        />

        {musicElementType === "intervals" ? (
          <IntervalSelector
            selectedIntervals={selectedIntervals}
            onIntervalToggle={onIntervalToggle}
            onSelectAll={onSelectAllIntervals}
            onClear={onClearIntervals}
          />
        ) : (
          <ChordSelector
            selectedChords={selectedChords}
            onChordToggle={onChordToggle}
            onSelectAll={onSelectAllChords}
            onClear={onClearChords}
          />
        )}

        <div className="setting-group">
          <div className="setting-actions">
            <button
              className="btn btn--primary"
              onClick={onGenerate}
              disabled={isGenerating || !canGenerate}
            >
              {isGenerating ? (
                <>
                  <LoadingIndicator size="small" />
                  <span>Генерація...</span>
                </>
              ) : (
                <>
                  <Volume2 />
                  <span>
                    {musicElementType === "intervals" 
                      ? "Згенерувати інтервали" 
                      : "Згенерувати акорди"}
                  </span>
                </>
              )}
            </button>

            {hasGenerated && (
              <button
                className="btn btn--ghost"
                onClick={onClearGenerated}
              >
                <Trash2 />
                <span>Очистити</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingSettings;
