import React, { useState, useRef, useEffect, useCallback } from "react";
import { Piano, Settings, Zap, VolumeX, ChevronDown } from "lucide-react";
import * as Tone from "tone";

const PianoKeyboard = ({ 
  isFixed = true, 
  showNoteNames = true, 
  showOctaves = true,
  startOctave = 2,   
  endOctave = 4,     
  onNotePlay = null 
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeKeys, setActiveKeys] = useState(new Set());
  const [showSettings, setShowSettings] = useState(false);
  const [pedalPressed, setPedalPressed] = useState(false);
  const [settings, setSettings] = useState({
    showNoteNames,
    showOctaves,
    startOctave: Math.min(startOctave, endOctave),
    endOctave: Math.max(startOctave, endOctave)
  });
    
  const samplerRef = useRef(null);
  const activeSoundsRef = useRef(new Map());
  const sustainedNotesRef = useRef(new Set());
  const pressedKeysRef = useRef(new Set());
  const isInitializedRef = useRef(false);

  const octaveNames = {
    '0': 'Субконтр',
    '1': 'Контр',
    '2': 'Велика',
    '3': 'Мала', 
    '4': 'Перша',
    '5': 'Друга',
    '6': 'Третя',
    '7': 'Четверта',
    '8': "П'ята"
  };

  const octaveOrder = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  const whiteKeys = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];
  const blackKeys = ['C#', 'D#', 'F#', 'G#', 'A#'];

  useEffect(() => {
    const initializeTone = async () => {
      if (isInitializedRef.current) return;
      
      try {
        const baseNotes = {
          'A0': 'https://tonejs.github.io/audio/salamander/A0.mp3',
          'C1': 'https://tonejs.github.io/audio/salamander/C1.mp3',
          'D#1': 'https://tonejs.github.io/audio/salamander/Ds1.mp3',
          'F#1': 'https://tonejs.github.io/audio/salamander/Fs1.mp3',
          'A1': 'https://tonejs.github.io/audio/salamander/A1.mp3',
          'C2': 'https://tonejs.github.io/audio/salamander/C2.mp3',
          'D#2': 'https://tonejs.github.io/audio/salamander/Ds2.mp3',
          'F#2': 'https://tonejs.github.io/audio/salamander/Fs2.mp3',
          'A2': 'https://tonejs.github.io/audio/salamander/A2.mp3',
          'C3': 'https://tonejs.github.io/audio/salamander/C3.mp3',
          'D#3': 'https://tonejs.github.io/audio/salamander/Ds3.mp3',
          'F#3': 'https://tonejs.github.io/audio/salamander/Fs3.mp3',
          'A3': 'https://tonejs.github.io/audio/salamander/A3.mp3',
          'C4': 'https://tonejs.github.io/audio/salamander/C4.mp3',
          'D#4': 'https://tonejs.github.io/audio/salamander/Ds4.mp3',
          'F#4': 'https://tonejs.github.io/audio/salamander/Fs4.mp3',
          'A4': 'https://tonejs.github.io/audio/salamander/A4.mp3',
          'C5': 'https://tonejs.github.io/audio/salamander/C5.mp3',
          'D#5': 'https://tonejs.github.io/audio/salamander/Ds5.mp3',
          'F#5': 'https://tonejs.github.io/audio/salamander/Fs5.mp3',
          'A5': 'https://tonejs.github.io/audio/salamander/A5.mp3',
          'C6': 'https://tonejs.github.io/audio/salamander/C6.mp3',
          'D#6': 'https://tonejs.github.io/audio/salamander/Ds6.mp3',
          'F#6': 'https://tonejs.github.io/audio/salamander/Fs6.mp3',
          'A6': 'https://tonejs.github.io/audio/salamander/A6.mp3',
          'C7': 'https://tonejs.github.io/audio/salamander/C7.mp3',
          'D#7': 'https://tonejs.github.io/audio/salamander/Ds7.mp3',
          'F#7': 'https://tonejs.github.io/audio/salamander/Fs7.mp3',
          'A7': 'https://tonejs.github.io/audio/salamander/A7.mp3',
          'C8': 'https://tonejs.github.io/audio/salamander/C8.mp3'
        };

        samplerRef.current = new Tone.Sampler({
          urls: baseNotes,
          release: 1,
          baseUrl: ""
        }).toDestination();

        await Tone.loaded();
        isInitializedRef.current = true;
        console.log('Piano sampler initialized successfully');
        
      } catch (error) {
        console.error('Error initializing Tone.js piano:', error);
        
        samplerRef.current = new Tone.PolySynth(Tone.Synth, {
          oscillator: {
            type: "triangle",
          },
          envelope: {
            attack: 0.01,
            decay: 0.2,
            sustain: 0.5,
            release: 1.5,
          },
        }).toDestination();
        
        isInitializedRef.current = true;
        console.log('Fallback synth initialized');
      }
    };

    const handleFirstInteraction = async () => {
      if (Tone.context.state !== 'running') {
        await Tone.start();
        console.log('Tone.js context started');
      }
      await initializeTone();
      document.removeEventListener('click', handleFirstInteraction);
      document.removeEventListener('touchstart', handleFirstInteraction);
    };

    document.addEventListener('click', handleFirstInteraction);
    document.addEventListener('touchstart', handleFirstInteraction);
    
    return () => {
      document.removeEventListener('click', handleFirstInteraction);
      document.removeEventListener('touchstart', handleFirstInteraction);
      
      stopAllNotes();
      
      if (samplerRef.current) {
        samplerRef.current.dispose();
      }
    };
  }, []);

  const getNotesForOctave = useCallback((octave, isBlack = false) => {
    if (octave === 0) {
      return isBlack ? ['A#'] : ['A', 'B'];
    } else if (octave === 8) {
      return isBlack ? [] : ['C'];
    }
    return isBlack ? blackKeys : whiteKeys;
  }, []);

  const getNormalizedRange = useCallback(() => {
    return { 
      start: Math.min(settings.startOctave, settings.endOctave), 
      end: Math.max(settings.startOctave, settings.endOctave) 
    };
  }, [settings.startOctave, settings.endOctave]);

  const stopNote = useCallback((note, octave, immediate = false) => {
    const noteKey = `${note}${octave}`;
    const activeSound = activeSoundsRef.current.get(noteKey);
    
    if (activeSound) {
      if (immediate && !pedalPressed) {
        try {
          if (samplerRef.current && samplerRef.current.triggerRelease) {
            samplerRef.current.triggerRelease(noteKey, Tone.now());
          }
        } catch (error) {
          console.warn('Error stopping note:', error);
        }
        
        clearTimeout(activeSound.timeout);
        activeSoundsRef.current.delete(noteKey);
        sustainedNotesRef.current.delete(noteKey);
      } else if (!pedalPressed) {
        try {
          if (samplerRef.current && samplerRef.current.triggerRelease) {
            samplerRef.current.triggerRelease(noteKey, Tone.now());
          }
        } catch (error) {
          console.warn('Error releasing note:', error);
        }
      } else {
        sustainedNotesRef.current.add(noteKey);
      }
    }
    
    setActiveKeys(prev => {
      const newSet = new Set(prev);
      newSet.delete(noteKey);
      return newSet;
    });
    
    pressedKeysRef.current.delete(noteKey);
  }, [pedalPressed]);

  const playNote = useCallback(async (note, octave, isHeld = false) => {
    const noteKey = `${note}${octave}`;
    
    if (!isInitializedRef.current || !samplerRef.current) {
      console.warn('Piano not initialized yet');
      return;
    }
    
    try {
      const existingSound = activeSoundsRef.current.get(noteKey);
      if (existingSound) {
        clearTimeout(existingSound.timeout);
        if (samplerRef.current.triggerRelease) {
          samplerRef.current.triggerRelease(noteKey, Tone.now());
        }
      }
      
      let duration;
      if (pedalPressed) {
        duration = 7;
      } else if (isHeld) {
        duration = 3.5;
      } else {
        duration = 1;
      }
      
      if (samplerRef.current.triggerAttack) {
        samplerRef.current.triggerAttack(noteKey, Tone.now());
      } else if (samplerRef.current.triggerAttackRelease) {
        samplerRef.current.triggerAttackRelease(noteKey, duration, Tone.now());
      }
      
      setActiveKeys(prev => new Set(prev).add(noteKey));

      const timeout = setTimeout(() => {
        if (!pedalPressed || !sustainedNotesRef.current.has(noteKey)) {
          try {
            if (samplerRef.current && samplerRef.current.triggerRelease) {
              samplerRef.current.triggerRelease(noteKey, Tone.now());
            }
          } catch (error) {
            console.warn('Error in timeout release:', error);
          }
          
          activeSoundsRef.current.delete(noteKey);
          setActiveKeys(prev => {
            const newSet = new Set(prev);
            newSet.delete(noteKey);
            return newSet;
          });
        }
      }, duration * 1000);
      
      activeSoundsRef.current.set(noteKey, { timeout, startTime: Tone.now() });
      
      if (onNotePlay) {
        const noteToSemitone = {
          'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4,
          'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2
        };
        const semitonesFromA4 = (octave - 4) * 12 + noteToSemitone[note];
        const frequency = 440 * Math.pow(2, semitonesFromA4 / 12);
        onNotePlay(note, octave, frequency);
      }
      
    } catch (error) {
      console.error('Error playing note:', error);
    }
  }, [pedalPressed, onNotePlay]);

  const handleKeyPress = useCallback((note, octave) => {
    const noteKey = `${note}${octave}`;
    pressedKeysRef.current.add(noteKey);
    playNote(note, octave, false);
  }, [playNote]);

  const handleKeyHold = useCallback((note, octave) => {
    const noteKey = `${note}${octave}`;
    pressedKeysRef.current.add(noteKey);
    playNote(note, octave, true);
  }, [playNote]);

  const handleKeyRelease = useCallback((note, octave) => {
    const noteKey = `${note}${octave}`;
    pressedKeysRef.current.delete(noteKey);
    
    if (!pedalPressed) {
      stopNote(note, octave, false);
    }
  }, [pedalPressed, stopNote]);

  const stopAllNotes = useCallback(() => {
    try {
      activeSoundsRef.current.forEach((sound) => {
        clearTimeout(sound.timeout);
      });
      
      if (samplerRef.current) {
        if (samplerRef.current.releaseAll) {
          samplerRef.current.releaseAll();
        } else if (samplerRef.current.triggerRelease) {
          activeSoundsRef.current.forEach((_, noteKey) => {
            try {
              samplerRef.current.triggerRelease(noteKey, Tone.now());
            } catch (error) {
              console.warn('Error releasing note:', noteKey, error);
            }
          });
        }
      }
      
      activeSoundsRef.current.clear();
      sustainedNotesRef.current.clear();
      setActiveKeys(new Set());
      pressedKeysRef.current.clear();
      
    } catch (error) {
      console.error('Error stopping all notes:', error);
    }
  }, []);

  const togglePedal = useCallback(() => {
    setPedalPressed(prev => {
      const newPedalState = !prev;
      
      if (!newPedalState) {
        stopAllNotes();
      }
      
      return newPedalState;
    });
  }, [stopAllNotes]);

  const getKeyPosition = useCallback((note, octave) => {
    const { start, end } = getNormalizedRange();
    
    let totalWhiteKeys = 0;
    let currentKeyIndex = 0;
    
    for (let oct = start; oct < octave; oct++) {
      const whiteKeysInOctave = getNotesForOctave(oct, false);
      totalWhiteKeys += whiteKeysInOctave.length;
    }
    
    const whiteKeysInCurrentOctave = getNotesForOctave(octave, false);
    if (whiteKeysInCurrentOctave.includes(note)) {
      currentKeyIndex = totalWhiteKeys + whiteKeysInCurrentOctave.indexOf(note);
    }
    
    let totalWhiteKeysCount = 0;
    for (let oct = start; oct <= end; oct++) {
      const whiteKeysInOctave = getNotesForOctave(oct, false);
      totalWhiteKeysCount += whiteKeysInOctave.length;
    }
    
    const whiteKeyWidth = 100 / totalWhiteKeysCount;
    
    if (whiteKeysInCurrentOctave.includes(note)) {
      return {
        left: `${currentKeyIndex * whiteKeyWidth}%`,
        width: `${whiteKeyWidth}%`
      };
    } else {
      const blackKeyPositions = {
        'C#': 0.7, 'D#': 1.7, 'F#': 3.7, 'G#': 4.7, 'A#': 5.7
      };
      
      let blackKeyOffset = 0;
      for (let oct = start; oct < octave; oct++) {
        const whiteKeysInOctave = getNotesForOctave(oct, false);
        blackKeyOffset += whiteKeysInOctave.length;
      }
      
      let intraOctaveOffset = 0;
      if (octave === 0 && note === 'A#') {
        intraOctaveOffset = 0.7;
      } else if (octave !== 0 && octave !== 8) {
        intraOctaveOffset = blackKeyPositions[note] || 0;
      }
      
      return {
        left: `${(blackKeyOffset + intraOctaveOffset) * whiteKeyWidth}%`,
        width: `${whiteKeyWidth * 0.6}%`
      };
    }
  }, [getNormalizedRange, getNotesForOctave]);

  const renderKeys = useCallback(() => {
    const keys = [];
    const { start, end } = getNormalizedRange();
    
    for (let octave = start; octave <= end; octave++) {
      const availableWhiteKeys = getNotesForOctave(octave, false);
      
      availableWhiteKeys.forEach(note => {
        const noteKey = `${note}${octave}`;
        const isActive = activeKeys.has(noteKey);
        const position = getKeyPosition(note, octave);
        
        keys.push(
          <button
            key={noteKey}
            className={`piano-key piano-key--white ${isActive ? 'piano-key--active' : ''}`}
            style={{
              left: position.left,
              width: position.width
            }}
            onMouseDown={() => handleKeyPress(note, octave)}
            onMouseUp={() => handleKeyRelease(note, octave)}
            onMouseLeave={() => handleKeyRelease(note, octave)}
            onTouchStart={(e) => {
              e.preventDefault();
              handleKeyHold(note, octave);
            }}
            onTouchEnd={(e) => {
              e.preventDefault();
              handleKeyRelease(note, octave);
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
    
    for (let octave = start; octave <= end; octave++) {
      const availableBlackKeys = getNotesForOctave(octave, true);
      
      availableBlackKeys.forEach(note => {
        const noteKey = `${note}${octave}`;
        const isActive = activeKeys.has(noteKey);
        const position = getKeyPosition(note, octave);
        
        keys.push(
          <button
            key={noteKey}
            className={`piano-key piano-key--black ${isActive ? 'piano-key--active' : ''}`}
            style={{
              left: position.left,
              width: position.width
            }}
            onMouseDown={() => handleKeyPress(note, octave)}
            onMouseUp={() => handleKeyRelease(note, octave)}
            onMouseLeave={() => handleKeyRelease(note, octave)}
            onTouchStart={(e) => {
              e.preventDefault();
              handleKeyHold(note, octave);
            }}
            onTouchEnd={(e) => {
              e.preventDefault();
              handleKeyRelease(note, octave);
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
  }, [getNormalizedRange, getNotesForOctave, activeKeys, settings, getKeyPosition, handleKeyPress, handleKeyHold, handleKeyRelease]);

  const updateSettings = useCallback((newSettings) => {
    setSettings(prev => {
      const updatedSettings = { ...prev, ...newSettings };
      
      if (newSettings.startOctave !== undefined || newSettings.endOctave !== undefined) {
        const normalizedStart = Math.min(updatedSettings.startOctave, updatedSettings.endOctave);
        const normalizedEnd = Math.max(updatedSettings.startOctave, updatedSettings.endOctave);
        
        return {
          ...updatedSettings,
          startOctave: normalizedStart,
          endOctave: normalizedEnd
        };
      }
      
      return updatedSettings;
    });
  }, []);

  const toggleExpanded = useCallback(() => {
    setIsExpanded(prev => !prev);
  }, []);

  return (
    <>
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
                onClick={() => setShowSettings(!showSettings)}
                title="Налаштування"
              >
                <Settings />
              </button>
              
              <button
                className={`piano-control-btn piano-control-btn--pedal ${pedalPressed ? 'piano-control-btn--pedal-active' : ''}`}
                onClick={togglePedal}
                title={pedalPressed ? "Відпустити педаль сустейну" : "Натиснути педаль сустейну"}
              >
                <Zap />
              </button>
              
              {activeKeys.size > 0 && (
                <button
                  className="piano-control-btn piano-control-btn--stop"
                  onClick={stopAllNotes}
                  title="Зупинити всі ноти"
                >
                  <VolumeX />
                </button>
              )}
              
              <button
                className="piano-control-btn"
                onClick={() => setIsExpanded(false)}
                title="Закрити клавіатуру"
              >
                <ChevronDown />
              </button>
            </div>
          </div>

          {showSettings && (
            <div className="piano-keyboard__settings">
              <div className="piano-setting">
                <label className="piano-setting__label">
                  <input
                    type="checkbox"
                    checked={settings.showNoteNames}
                    onChange={(e) => updateSettings({ showNoteNames: e.target.checked })}
                  />
                  Показувати назви нот
                </label>
              </div>
              
              <div className="piano-setting">
                <label className="piano-setting__label">
                  <input
                    type="checkbox"
                    checked={settings.showOctaves}
                    onChange={(e) => updateSettings({ showOctaves: e.target.checked })}
                  />
                  Показувати октави
                </label>
              </div>
              
              <div className="piano-setting">
                <label className="piano-setting__label">
                  Початкова октава:
                  <select
                    value={settings.startOctave}
                    onChange={(e) => updateSettings({ startOctave: parseInt(e.target.value) })}
                  >
                    {octaveOrder.map((value) => (
                      <option key={value} value={value}>{octaveNames[value]} ({value})</option>
                    ))}
                  </select>
                </label>
              </div>
              
              <div className="piano-setting">
                <label className="piano-setting__label">
                  Кінцева октава:
                  <select
                    value={settings.endOctave}
                    onChange={(e) => updateSettings({ endOctave: parseInt(e.target.value) })}
                  >
                    {octaveOrder.map((value) => (
                      <option key={value} value={value}>{octaveNames[value]} ({value})</option>
                    ))}
                  </select>
                </label>
              </div>
            </div>
          )}

          <div className="piano-keyboard__keys">
            {renderKeys()}
          </div>
        </div>
      )}
    </>
  );
};

export default PianoKeyboard;