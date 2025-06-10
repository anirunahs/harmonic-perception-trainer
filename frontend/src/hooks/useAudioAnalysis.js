import { useState, useCallback } from 'react';
import api from '../api';

export const useAudioAnalysis = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState(null);

  const analyzeAudio = useCallback(async (audioData, analysisType = 'frequency', sampleRate = 44100) => {
    setIsAnalyzing(true);
    setError(null);
    setAnalysisResult(null);

    try {
      let base64Audio = audioData;
      
      if (audioData instanceof Blob) {
        base64Audio = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => {
            const result = reader.result.split(',')[1];
            resolve(result);
          };
          reader.readAsDataURL(audioData);
        });
      }

      const response = await api.post('/api/audio/analyze/', {
        audio_data: base64Audio,
        sample_rate: sampleRate,
        analysis_type: analysisType
      });

      setAnalysisResult(response.data);
      return response.data;
    } catch (err) {
      const errorMessage = err.response?.data?.error || 'Помилка аналізу аудіо';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const analyzeFrequency = useCallback(async (audioData, sampleRate = 44100) => {
    return analyzeAudio(audioData, 'frequency', sampleRate);
  }, [analyzeAudio]);

  const reduceNoise = useCallback(async (audioData, sampleRate = 44100) => {
    return analyzeAudio(audioData, 'noise_reduction', sampleRate);
  }, [analyzeAudio]);

  const analyzeInterval = useCallback(async (audioData, sampleRate = 44100) => {
    return analyzeAudio(audioData, 'interval', sampleRate);
  }, [analyzeAudio]);

  const clearResults = useCallback(() => {
    setAnalysisResult(null);
    setError(null);
  }, []);

  const frequencyToNote = useCallback((frequency) => {
    if (!frequency || frequency <= 0) return null;
    
    const A4 = 440.0;
    const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    
    const semitonesFromA4 = Math.round(12 * Math.log2(frequency / A4));
    
    const octave = 4 + Math.floor(semitonesFromA4 / 12);
    const noteIndex = (9 + semitonesFromA4) % 12;
    
    return `${notes[noteIndex]}${octave}`;
  }, []);

  const noteToFrequency = useCallback((note) => {
    if (!note) return null;
    
    const A4 = 440.0;
    const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    
    const noteMatch = note.match(/^([A-G]#?)(\d+)$/);
    if (!noteMatch) return null;
    
    const [, noteName, octaveStr] = noteMatch;
    const octave = parseInt(octaveStr);
    
    const noteIndex = notes.indexOf(noteName);
    if (noteIndex === -1) return null;
    
    const semitonesFromA4 = (octave - 4) * 12 + (noteIndex - 9);
    
    return A4 * Math.pow(2, semitonesFromA4 / 12);
  }, []);

  const compareFrequencies = useCallback((freq1, freq2, tolerance = 20) => {
    if (!freq1 || !freq2) return { match: false, difference: null };
    
    const difference = Math.abs(freq1 - freq2);
    const match = difference <= tolerance;
    
    return { match, difference };
  }, []);

  return {
    isAnalyzing,
    analysisResult,
    error,
    
    analyzeAudio,
    analyzeFrequency,
    reduceNoise,
    analyzeInterval,
    
    frequencyToNote,
    noteToFrequency,
    compareFrequencies,
    clearResults,
    
    setError
  };
};