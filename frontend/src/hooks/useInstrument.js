import { useState, useEffect, useRef, useCallback } from 'react';
import * as Tone from 'tone';

/**
 * Instrument configurations with sample URLs
 * Using free, high-quality samples
 */
const INSTRUMENT_CONFIGS = {
  piano: {
    name: 'Piano',
    // Salamander Grand Piano samples (public domain)
    baseUrl: 'https://tonejs.github.io/audio/salamander/',
    samples: {
      'A0': 'A0.mp3',
      'C1': 'C1.mp3',
      'D#1': 'Ds1.mp3',
      'F#1': 'Fs1.mp3',
      'A1': 'A1.mp3',
      'C2': 'C2.mp3',
      'D#2': 'Ds2.mp3',
      'F#2': 'Fs2.mp3',
      'A2': 'A2.mp3',
      'C3': 'C3.mp3',
      'D#3': 'Ds3.mp3',
      'F#3': 'Fs3.mp3',
      'A3': 'A3.mp3',
      'C4': 'C4.mp3',
      'D#4': 'Ds4.mp3',
      'F#4': 'Fs4.mp3',
      'A4': 'A4.mp3',
      'C5': 'C5.mp3',
      'D#5': 'Ds5.mp3',
      'F#5': 'Fs5.mp3',
      'A5': 'A5.mp3',
      'C6': 'C6.mp3',
      'D#6': 'Ds6.mp3',
      'F#6': 'Fs6.mp3',
      'A6': 'A6.mp3',
      'C7': 'C7.mp3',
      'D#7': 'Ds7.mp3',
      'F#7': 'Fs7.mp3',
      'A7': 'A7.mp3',
      'C8': 'C8.mp3',
    },
    options: {
      release: 1.5,
      volume: -6,
    }
  },
  guitar: {
    name: 'Guitar',
    // Nylon acoustic guitar samples from MIDI.js SoundFonts (FluidR3_GM)
    // GitHub: gleitz/midi-js-soundfonts - uses Eb/Gb notation
    baseUrl: 'https://gleitz.github.io/midi-js-soundfonts/FluidR3_GM/acoustic_guitar_nylon-mp3/',
    samples: {
      'E2': 'E2.mp3',
      'G2': 'G2.mp3',
      'A2': 'A2.mp3',
      'C3': 'C3.mp3',
      'E3': 'E3.mp3',
      'G3': 'G3.mp3',
      'A3': 'A3.mp3',
      'C4': 'C4.mp3',
      'E4': 'E4.mp3',
      'G4': 'G4.mp3',
      'A4': 'A4.mp3',
      'C5': 'C5.mp3',
      'E5': 'E5.mp3',
      'G5': 'G5.mp3',
      'A5': 'A5.mp3',
      'C6': 'C6.mp3',
    },
    options: {
      release: 1.2,
      volume: -3,
    }
  }
};

/**
 * Interval semitones mapping
 */
const INTERVALS = {
  'minor_second': 1,
  'major_second': 2,
  'minor_third': 3,
  'major_third': 4,
  'perfect_fourth': 5,
  'tritone': 6,
  'perfect_fifth': 7,
  'minor_sixth': 8,
  'major_sixth': 9,
  'minor_seventh': 10,
  'major_seventh': 11,
  'perfect_octave': 12,
};

/**
 * Hook for playing instruments with high-quality samples
 */
export const useInstrument = (instrumentType = 'piano') => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentInstrument, setCurrentInstrument] = useState(instrumentType);
  
  const instrumentRef = useRef(null);
  const reverbRef = useRef(null);

  /**
   * Initialize or change instrument
   */
  const loadInstrument = useCallback(async (type) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Dispose old instrument
      if (instrumentRef.current) {
        instrumentRef.current.dispose();
      }
      
      // Initialize reverb for natural sound
      if (!reverbRef.current) {
        reverbRef.current = new Tone.Reverb({
          decay: 2.5,
          wet: 0.2,
        }).toDestination();
        await reverbRef.current.generate();
      }
      
      const config = INSTRUMENT_CONFIGS[type];
      
      if (!config) {
        throw new Error(`Unknown instrument: ${type}`);
      }
      
      // Create sampler with real samples
      instrumentRef.current = new Tone.Sampler({
        urls: config.samples,
        baseUrl: config.baseUrl,
        ...config.options,
        onload: () => {
          console.log(`${config.name} samples loaded`);
        },
        onerror: (err) => {
          console.error('Sample loading error:', err);
          setError(`Failed to load ${config.name} samples`);
        }
      }).connect(reverbRef.current);
      
      // Wait for samples to load
      await Tone.loaded();
      
      setCurrentInstrument(type);
      setIsLoaded(true);
      
    } catch (err) {
      console.error('Instrument loading error:', err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Initialize on mount and when instrument type changes
   */
  useEffect(() => {
    loadInstrument(instrumentType);
    
    return () => {
      if (instrumentRef.current) {
        instrumentRef.current.dispose();
      }
    };
  }, [instrumentType, loadInstrument]);

  /**
   * Play a single note
   */
  const playNote = useCallback(async (note, duration = '2n') => {
    if (!instrumentRef.current || !isLoaded) {
      console.warn('Instrument not loaded');
      return;
    }
    
    // Start audio context if needed
    if (Tone.context.state !== 'running') {
      await Tone.start();
    }
    
    instrumentRef.current.triggerAttackRelease(note, duration);
  }, [isLoaded]);

  /**
   * Play harmonic interval (two notes together)
   */
  const playHarmonicInterval = useCallback(async (baseNote, intervalType, duration = '2n') => {
    if (!instrumentRef.current || !isLoaded) {
      console.warn('Instrument not loaded');
      return;
    }
    
    if (Tone.context.state !== 'running') {
      await Tone.start();
    }
    
    const semitones = INTERVALS[intervalType];
    if (semitones === undefined) {
      console.error('Unknown interval:', intervalType);
      return;
    }
    
    // Calculate target note
    const targetNote = Tone.Frequency(baseNote).transpose(semitones).toNote();
    
    // Play both notes simultaneously
    instrumentRef.current.triggerAttackRelease([baseNote, targetNote], duration);
    
    return { baseNote, targetNote, semitones };
  }, [isLoaded]);

  /**
   * Play melodic interval (two notes in sequence)
   */
  const playMelodicInterval = useCallback(async (baseNote, intervalType, noteDuration = '2n') => {
    if (!instrumentRef.current || !isLoaded) {
      console.warn('Instrument not loaded');
      return;
    }
    
    if (Tone.context.state !== 'running') {
      await Tone.start();
    }
    
    const semitones = INTERVALS[intervalType];
    if (semitones === undefined) {
      console.error('Unknown interval:', intervalType);
      return;
    }
    
    const targetNote = Tone.Frequency(baseNote).transpose(semitones).toNote();
    
    // Play first note
    instrumentRef.current.triggerAttackRelease(baseNote, noteDuration);
    
    // Schedule second note after first finishes
    const durationSeconds = Tone.Time(noteDuration).toSeconds();
    
    setTimeout(() => {
      if (instrumentRef.current) {
        instrumentRef.current.triggerAttackRelease(targetNote, noteDuration);
      }
    }, durationSeconds * 1000);
    
    return { baseNote, targetNote, semitones };
  }, [isLoaded]);

  /**
   * Stop all playing sounds
   */
  const stopAll = useCallback(() => {
    if (instrumentRef.current && instrumentRef.current.releaseAll) {
      instrumentRef.current.releaseAll();
    }
  }, []);

  /**
   * Change instrument
   */
  const changeInstrument = useCallback((type) => {
    if (type !== currentInstrument) {
      setIsLoaded(false);
      loadInstrument(type);
    }
  }, [currentInstrument, loadInstrument]);

  return {
    isLoaded,
    isLoading,
    error,
    currentInstrument,
    playNote,
    playHarmonicInterval,
    playMelodicInterval,
    stopAll,
    changeInstrument,
    availableInstruments: Object.keys(INSTRUMENT_CONFIGS),
  };
};

export default useInstrument;
