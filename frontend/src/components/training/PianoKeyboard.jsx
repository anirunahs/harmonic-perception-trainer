import React, { useState, useRef, useEffect, useCallback } from "react";
import { Piano, Settings, ChevronDown } from "lucide-react";

const PianoKeyboard = ({ 
  isFixed = true, 
  showNoteNames = true, 
  showOctaves = true,
  startOctave = 2,   
  endOctave = 4,     
  onNotePlay = null 
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [settings, setSettings] = useState({
    showNoteNames,
    showOctaves,
    startOctave: Math.min(startOctave, endOctave),
    endOctave: Math.max(startOctave, endOctave)
  });
  
  const audioContextRef = useRef(null);

  // Октави та їх назви
  const octaveNames = {
    '-1': 'Субконтр',
    '0': 'Контр',
    '1': 'Велика',
    '2': 'Мала', 
    '3': 'Перша',
    '4': 'Друга',
    '5': 'Третя',
    '6': 'Четверта',
    '7': "П'ята"
  };

  const octaveOrder = [-1, 0, 1, 2, 3, 4, 5, 6, 7];
  const whiteKeys = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
  const blackKeys = ['C#', 'D#', 'F#', 'G#', 'A#'];

  // Спеціальні ноти для різних октав
  const getNotesForOctave = useCallback((octave, isBlack = false) => {
    if (octave === -1) {
      return isBlack ? ['A#'] : ['A', 'B'];
    } else if (octave === 7) {
      return isBlack ? [] : ['C'];
    }
    return isBlack ? blackKeys : whiteKeys;
  }, []);

  // Нормалізація діапазону октав
  const getNormalizedRange = useCallback(() => {
    return { 
      start: Math.min(settings.startOctave, settings.endOctave), 
      end: Math.max(settings.startOctave, settings.endOctave) 
    };
  }, [settings.startOctave, settings.endOctave]);

  // Ініціалізація AudioContext
  useEffect(() => {
    const initAudio = () => {
      if (!audioContextRef.current) {
        try {
          audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
        } catch (error) {
          console.error('Помилка створення AudioContext:', error);
        }
      }
    };

    const handleFirstInteraction = () => {
      initAudio();
      document.removeEventListener('click', handleFirstInteraction);
      document.removeEventListener('touchstart', handleFirstInteraction);
    };

    document.addEventListener('click', handleFirstInteraction);
    document.addEventListener('touchstart', handleFirstInteraction);
    
    return () => {
      document.removeEventListener('click', handleFirstInteraction);
      document.removeEventListener('touchstart', handleFirstInteraction);
      
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
    };
  }, []);

  // Позиціонування клавіш
  const getKeyPosition = useCallback((note, octave) => {
    const { start, end } = getNormalizedRange();
    
    let totalWhiteKeys = 0;
    let currentKeyIndex = 0;
    
    // Рахуємо білі клавіші до поточної октави
    for (let oct = start; oct < octave; oct++) {
      const whiteKeysInOctave = getNotesForOctave(oct, false);
      totalWhiteKeys += whiteKeysInOctave.length;
    }
    
    const whiteKeysInCurrentOctave = getNotesForOctave(octave, false);
    if (whiteKeysInCurrentOctave.includes(note)) {
      currentKeyIndex = totalWhiteKeys + whiteKeysInCurrentOctave.indexOf(note);
    }
    
    // Загальна кількість білих клавіш
    let totalWhiteKeysCount = 0;
    for (let oct = start; oct <= end; oct++) {
      const whiteKeysInOctave = getNotesForOctave(oct, false);
      totalWhiteKeysCount += whiteKeysInOctave.length;
    }
    
    const whiteKeyWidth = 100 / totalWhiteKeysCount;
    
    if (whiteKeysInCurrentOctave.includes(note)) {
      // Біла клавіша
      return {
        left: `${currentKeyIndex * whiteKeyWidth}%`,
        width: `${whiteKeyWidth}%`
      };
    } else {
      // Чорна клавіша
      const blackKeyPositions = {
        'C#': 0.7, 'D#': 1.7, 'F#': 3.7, 'G#': 4.7, 'A#': 5.7
      };
      
      let blackKeyOffset = 0;
      for (let oct = start; oct < octave; oct++) {
        const whiteKeysInOctave = getNotesForOctave(oct, false);
        blackKeyOffset += whiteKeysInOctave.length;
      }
      
      let intraOctaveOffset = 0;
      if (octave === -1 && note === 'A#') {
        intraOctaveOffset = 0.7;
      } else if (octave !== -1 && octave !== 7) {
        intraOctaveOffset = blackKeyPositions[note] || 0;
      }
      
      return {
        left: `${(blackKeyOffset + intraOctaveOffset) * whiteKeyWidth}%`,
        width: `${whiteKeyWidth * 0.6}%`
      };
    }
  }, [getNormalizedRange, getNotesForOctave]);

  // Рендер клавіш
  const renderKeys = useCallback(() => {
    const keys = [];
    const { start, end } = getNormalizedRange();
    
    // Білі клавіші
    for (let octave = start; octave <= end; octave++) {
      const availableWhiteKeys = getNotesForOctave(octave, false);
      
      availableWhiteKeys.forEach(note => {
        const noteKey = `${note}${octave}`;
        const position = getKeyPosition(note, octave);
        
        keys.push(
          <button
            key={noteKey}
            className="piano-key piano-key--white"
            style={{
              left: position.left,
              width: position.width
            }}
          >
            {(settings.showNoteNames || settings.showOctaves) && (
              <span className="piano-key__label">
                {settings.showNoteNames && note}
                {settings.showOctaves && (
                  <span className="piano-key__octave">{octave}</span>
                )}
              </span>
            )}
          </button>
        );
      });
    }
    
    // Чорні клавіші
    for (let octave = start; octave <= end; octave++) {
      const availableBlackKeys = getNotesForOctave(octave, true);
      
      availableBlackKeys.forEach(note => {
        const noteKey = `${note}${octave}`;
        const position = getKeyPosition(note, octave);
        
        keys.push(
          <button
            key={noteKey}
            className="piano-key piano-key--black"
            style={{
              left: position.left,
              width: position.width
            }}
          >
            {(settings.showNoteNames || settings.showOctaves) && (
              <span className="piano-key__label piano-key__label--black">
                {settings.showNoteNames && note}
                {settings.showOctaves && (
                  <span className="piano-key__octave">{octave}</span>
                )}
              </span>
            )}
          </button>
        );
      });
    }
    
    return keys;
  }, [getNormalizedRange, getNotesForOctave, settings, getKeyPosition]);

  const toggleExpanded = useCallback(() => {
    setIsExpanded(prev => !prev);
  }, []);

  return (
    <>
      {/* Кнопка відкриття коли панель схована */}
      {!isExpanded && (
        <div className="piano-keyboard-toggle piano-keyboard-toggle--fixed">
          <button
            className="piano-toggle-btn"
            onClick={() => setIsExpanded(true)}
            title="Відкрити фортепіано"
          >
            <Piano />
            <span>Фортепіано</span>
          </button>
        </div>
      )}
      
      {/* Основна панель фортепіано */}
      {isExpanded && (
        <div className={`piano-keyboard ${isFixed ? 'piano-keyboard--fixed' : ''} piano-keyboard--expanded`}>
          <div className="piano-keyboard__header" onClick={() => setIsExpanded(false)}>
            <div className="piano-keyboard__info">
              <Piano className="piano-keyboard__icon" />
              <span className="piano-keyboard__title">Фортепіано</span>
              <span className="piano-keyboard__range">
                {octaveNames[getNormalizedRange().start]} - {octaveNames[getNormalizedRange().end]}
              </span>
            </div>
            
            <div className="piano-keyboard__controls" onClick={(e) => e.stopPropagation()}>
              <button
                className="piano-control-btn"
                onClick={() => setIsExpanded(false)}
                title="Закрити клавіатуру"
              >
                <ChevronDown />
              </button>
            </div>
          </div>

          <div className="piano-keyboard__keys">
            {renderKeys()}
          </div>
        </div>
      )}
    </>
  );
};

export default PianoKeyboard;