import React from "react";
import { Settings, Volume2, Trash2 } from "lucide-react";
import LoadingIndicator from "../LoadingIndicator";
import NoteSelector from "./NoteSelector";
import IntervalSelector from "./IntervalSelector";

const TrainingSettings = ({
  selectedNote,
  setSelectedNote,
  selectedIntervals,
  onIntervalToggle,
  onSelectAllIntervals,
  onClearIntervals,
  onGenerateIntervals,
  onClearGenerated,
  isGenerating,
  hasGeneratedIntervals
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
        <NoteSelector
          selectedNote={selectedNote}
          onNoteSelect={setSelectedNote}
        />

        <IntervalSelector
          selectedIntervals={selectedIntervals}
          onIntervalToggle={onIntervalToggle}
          onSelectAll={onSelectAllIntervals}
          onClear={onClearIntervals}
        />

        <div className="setting-group">
          <div className="setting-actions">
            <button
              className="btn btn--primary"
              onClick={onGenerateIntervals}
              disabled={isGenerating || selectedIntervals.length === 0}
            >
              {isGenerating ? (
                <>
                  <LoadingIndicator size="small" />
                  <span>Генерація...</span>
                </>
              ) : (
                <>
                  <Volume2 />
                  <span>Згенерувати інтервали</span>
                </>
              )}
            </button>

            {hasGeneratedIntervals && (
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