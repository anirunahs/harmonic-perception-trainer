import React, { useState, useRef, useEffect, useCallback } from "react";
import { Piano, ChevronDown } from "lucide-react";

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
  const oscillatorsRef = useRef(new Map());
  const timeoutsRef = useRef(new Map());

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

  const getNotesForOctave = useCallback((octave, isBlack = false) => {
    if (octave === -1) {
      return isBlack ? ['A#'] : ['A', 'B'];
    } else if (octave === 7) {
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

  const getFrequency = useCallback((note, octave) => {
    const noteToSemitone = {
      'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4,
      'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2
    };
    
    const semitonesFromA4 = (octave - 4) * 12 + noteToSemitone[note];
    return 440 * Math.pow(2, semitonesFromA4 / 12);
  }, []);

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
      
      stopAllNotes();
      
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
    };
  }, []);

  const generatePianoTone = useCallback((frequency, duration = 1.0) => {
    if (!audioContextRef.current || audioContextRef.current.state !== 'running') {
      return null;
    }

    try {
      const audioContext = audioContextRef.current;
      const gainNode = audioContext.createGain();
      
      const oscillator = audioContext.createOscillator();
      oscillator.type = 'triangle';
      oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime);
      
      const harmonics = [
        { freq: frequency * 2, gain: 0.3, type: 'sine' },
        { freq: frequency * 3, gain: 0.15, type: 'triangle' },
        { freq: frequency * 4, gain: 0.1, type: 'sine' },
        { freq: frequency * 5, gain: 0.06, type: 'triangle' }
      ];
      
      const harmonicNodes = [];
      
      harmonics.forEach(harmonic => {
        try {
          const osc = audioContext.createOscillator();
          const harmonicGain = audioContext.createGain();
          
          osc.type = harmonic.type;
          osc.frequency.setValueAtTime(harmonic.freq, audioContext.currentTime);
          harmonicGain.gain.setValueAtTime(harmonic.gain, audioContext.currentTime);
          
          osc.connect(harmonicGain);
          harmonicGain.connect(gainNode);
          osc.start();
          
          harmonicNodes.push({ oscillator: osc, gain: harmonicGain });
        } catch (error) {
          console.warn('Помилка створення гармоніки:', error);
        }
      });

      const now = audioContext.currentTime;
      const attackTime = 0.01;
      const decayTime = 0.1;
      const sustainLevel = 0.5;
      const releaseTime = duration - attackTime - decayTime;
      
      gainNode.gain.setValueAtTime(0, now);
      gainNode.gain.linearRampToValueAtTime(0.8, now + attackTime);
      gainNode.gain.exponentialRampToValueAtTime(sustainLevel, now + attackTime + decayTime);
      gainNode.gain.exponentialRampToValueAtTime(0.001, now + duration);
      
      gainNode.connect(audioContext.destination);
      oscillator.connect(gainNode);
      oscillator.start();
      oscillator.stop(now + duration);
      
      harmonicNodes.forEach(node => {
        node.oscillator.stop(now + duration);
      });
      
      return { 
        oscillator, 
        harmonicNodes, 
        gainNode,
        duration,
        startTime: now
      };
      
    } catch (error) {
      console.error('Помилка генерації звуку:', error);
      return null;
    }
  }, []);

  const playNote = useCallback(async (note, octave) => {
    const noteKey = `${note}${octave}`;
    
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }
      
      const existingAudio = oscillatorsRef.current.get(noteKey);
      if (existingAudio) {
        try {
          const now = audioContextRef.current.currentTime;
          existingAudio.gainNode.gain.cancelScheduledValues(now);
          existingAudio.gainNode.gain.exponentialRampToValueAtTime(0.001, now + 0.01);
          
          setTimeout(() => {
            try {
              if (existingAudio.oscillator.playbackState !== 'finished') {
                existingAudio.oscillator.stop();
              }
              existingAudio.harmonicNodes.forEach(node => {
                if (node && node.oscillator && node.oscillator.playbackState !== 'finished') {
                  node.oscillator.stop();
                }
              });
            } catch (error) { }
          }, 10);
        } catch (error) {
          console.warn('Помилка плавного переривання:', error);
        }
      }
      
      const frequency = getFrequency(note, octave);
      const duration = 1;
      
      const noteAudio = generatePianoTone(frequency, duration);
      
      if (noteAudio) {
        oscillatorsRef.current.set(noteKey, noteAudio);
        
        const timeout = setTimeout(() => {
          oscillatorsRef.current.delete(noteKey);
          timeoutsRef.current.delete(noteKey);
        }, duration * 1000);
        
        timeoutsRef.current.set(noteKey, timeout);
        
        if (onNotePlay) {
          onNotePlay(note, octave, frequency);
        }
      }
    } catch (error) {
      console.error('Помилка відтворення ноти:', error);
    }
  }, [getFrequency, generatePianoTone, onNotePlay]);

  const stopAllNotes = useCallback(() => {
    try {
      timeoutsRef.current.forEach(timeout => clearTimeout(timeout));
      timeoutsRef.current.clear();
      
      oscillatorsRef.current.forEach((noteAudio, noteKey) => {
        try {
          const now = audioContextRef.current?.currentTime || 0;
          noteAudio.gainNode.gain.cancelScheduledValues(now);
          noteAudio.gainNode.gain.exponentialRampToValueAtTime(0.001, now + 0.05);
          
          setTimeout(() => {
            try {
              if (noteAudio.oscillator.playbackState !== 'finished') {
                noteAudio.oscillator.stop();
              }
              noteAudio.harmonicNodes.forEach(node => {
                if (node && node.oscillator && node.oscillator.playbackState !== 'finished') {
                  node.oscillator.stop();
                }
              });
            } catch (error) { }
          }, 50);
        } catch (error) {
          console.warn('Помилка зупинки ноти:', error);
        }
      });
      
      oscillatorsRef.current.clear();
      
    } catch (error) {
      console.error('Помилка зупинки всіх нот:', error);
    }
  }, []);

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

  const renderKeys = useCallback(() => {
    const keys = [];
    const { start, end } = getNormalizedRange();
    
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
            onClick={() => playNote(note, octave)}
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
        const position = getKeyPosition(note, octave);
        
        keys.push(
          <button
            key={noteKey}
            className="piano-key piano-key--black"
            style={{
              left: position.left,
              width: position.width
            }}
            onClick={() => playNote(note, octave)}
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
  }, [getNormalizedRange, getNotesForOctave, settings, getKeyPosition, playNote]);

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