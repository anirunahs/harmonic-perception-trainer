import React, { useState, useCallback } from "react";
import { Settings, Mic, Play, Save, RefreshCw } from "lucide-react";
import MicrophoneRecorder from "../common/MicrophoneRecorder";
import LoadingIndicator from "../LoadingIndicator";
import api from "../../api";

const VocalRangeSetup = ({ isOpen, onClose, onSave }) => {
  const [step, setStep] = useState(1);
  const [minNote, setMinNote] = useState(null);
  const [maxNote, setMaxNote] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  const analyzeAudioFrequency = useCallback(async (audioBlob) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = async (event) => {
        try {
          const arrayBuffer = event.target.result;
          const audioContext = new (window.AudioContext || window.webkitAudioContext)();
          const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
          
          const channelData = audioBuffer.getChannelData(0);
          const sampleRate = audioBuffer.sampleRate;
          
          // Вікно Хемінга для зменшення шуму
          const windowSize = Math.min(4096, channelData.length);
          const windowed = new Float32Array(windowSize);
          
          for (let i = 0; i < windowSize; i++) {
            const windowValue = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (windowSize - 1));
            windowed[i] = channelData[i] * windowValue;
          }
          
          // FFT 
          const fftSize = windowSize;
          const fft = new Float32Array(fftSize);
          
          // Автокореляція для пошуку основної частоти
          let maxCorrelation = 0;
          let bestPeriod = 0;
          
          const minPeriod = Math.floor(sampleRate / 800);
          const maxPeriod = Math.floor(sampleRate / 50);
          
          for (let period = minPeriod; period < maxPeriod && period < windowSize / 2; period++) {
            let correlation = 0;
            for (let i = 0; i < windowSize - period; i++) {
              correlation += windowed[i] * windowed[i + period];
            }
            
            if (correlation > maxCorrelation) {
              maxCorrelation = correlation;
              bestPeriod = period;
            }
          }
          
          const frequency = bestPeriod > 0 ? sampleRate / bestPeriod : 0;
          
          // Конвертація частоти в ноту
          const noteInfo = frequencyToNote(frequency);
          
          resolve({
            frequency: frequency,
            note: noteInfo.note,
            octave: noteInfo.octave,
            cents: noteInfo.cents
          });
          
        } catch (error) {
          reject(error);
        }
      };
      reader.onerror = reject;
      reader.readAsArrayBuffer(audioBlob);
    });
  }, []);

  // Конвертація
  const frequencyToNote = (frequency) => {
    const A4 = 440.0;
    const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    
    if (frequency <= 0) return { note: '?', octave: 0, cents: 0 };
    
    const semitonesFromA4 = 12 * Math.log2(frequency / A4);
    const octave = Math.floor((semitonesFromA4 + 57) / 12);
    const noteIndex = Math.round(semitonesFromA4 + 57) % 12;
    const cents = Math.round((semitonesFromA4 + 57 - Math.round(semitonesFromA4 + 57)) * 100);
    
    return {
      note: notes[noteIndex],
      octave: octave,
      cents: cents
    };
  };

  const handleRecordingComplete = async (audioBlob) => {
    setIsProcessing(true);
    setError(null);
    
    try {
      const analysis = await analyzeAudioFrequency(audioBlob);
      
      if (analysis.frequency < 50 || analysis.frequency > 2000) {
        setError("Не вдалося визначити частоту. Спробуйте заспівати гучніше та чистіше.");
        return;
      }
      
      const noteData = {
        frequency: analysis.frequency,
        note: analysis.note,
        octave: analysis.octave,
        noteName: `${analysis.note}${analysis.octave}`
      };
      
      if (step === 2) {
        setMinNote(noteData);
        setStep(3);
      } else if (step === 3) {
        if (noteData.frequency <= minNote.frequency) {
          setError("Максимальна нота має бути вище мінімальної. Спробуйте заспівати вище.");
          return;
        }
        setMaxNote(noteData);
        setStep(4);
      }
      
    } catch (error) {
      console.error('Error analyzing audio:', error);
      setError("Помилка аналізу аудіо. Спробуйте ще раз.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSave = async () => {
    setIsProcessing(true);
    setError(null);
    
    try {
      await api.post('/api/user/vocal-range/', {
        min_frequency: minNote.frequency,
        max_frequency: maxNote.frequency,
        min_note: minNote.noteName,
        max_note: maxNote.noteName
      });
      
      if (onSave) {
        onSave({
          minNote: minNote,
          maxNote: maxNote
        });
      }
      
      onClose();
    } catch (error) {
      console.error('Error saving vocal range:', error);
      setError("Помилка збереження. Спробуйте ще раз.");
    } finally {
      setIsProcessing(false);
    }
  };

  const resetSetup = () => {
    setStep(1);
    setMinNote(null);
    setMaxNote(null);
    setError(null);
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-content vocal-range-setup">
        <div className="modal-header">
          <h2 className="modal-title">
            <Settings />
            Налаштування вокального діапазону
          </h2>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>

        <div className="modal-body">
          {step === 1 && (
            <div className="setup-step">
              <div className="step-icon">
                <Mic />
              </div>
              <h3>Ласкаво просимо в налаштування вокального діапазону!</h3>
              <p>
                Ми визначимо найнижчу та найвищу ноту, яку ви можете комфортно заспівати. 
                Це допоможе створювати тести, які підходять саме для вашого голосу.
              </p>
              <div className="step-instructions">
                <h4>Як це працює:</h4>
                <ol>
                  <li>Спочатку заспівайте найнижчу ноту, яку можете</li>
                  <li>Потім заспійте найвищу ноту, яку можете</li>
                  <li>Ми збережемо ваш діапазон для майбутніх тестів</li>
                </ol>
              </div>
              <button className="btn btn--primary" onClick={() => setStep(2)}>
                Почати налаштування
              </button>
            </div>
          )}

          {step === 2 && (
            <div className="setup-step">
              <div className="step-icon">
                <Mic />
              </div>
              <h3>Крок 1: Мінімальна нота</h3>
              <p>Заспівайте найнижчу ноту, яку можете комфортно взяти</p>
              
              {error && (
                <div className="alert alert--error">
                  {error}
                </div>
              )}
              
              {isProcessing && (
                <div className="processing-state">
                  <LoadingIndicator />
                  <span>Аналізуємо голос...</span>
                </div>
              )}
              
              <MicrophoneRecorder
                onRecordingComplete={handleRecordingComplete}
                maxDuration={5}
                autoAnalyze={true}
                showPlayback={false}
              />
              
              <div className="step-controls">
                <button className="btn btn--ghost" onClick={() => setStep(1)}>
                  Назад
                </button>
              </div>
            </div>
          )}

          {step === 3 && (
            <div className="setup-step">
              <div className="step-icon">
                <Mic />
              </div>
              <h3>Крок 2: Максимальна нота</h3>
              <p>Заспівайте найвищу ноту, яку можете комфортно взяти</p>
              
              <div className="current-range">
                <div className="note-display">
                  <strong>Мінімальна нота:</strong> {minNote?.noteName} ({minNote?.frequency.toFixed(1)} Гц)
                </div>
              </div>
              
              {error && (
                <div className="alert alert--error">
                  {error}
                </div>
              )}
              
              {isProcessing && (
                <div className="processing-state">
                  <LoadingIndicator />
                  <span>Аналізуємо голос...</span>
                </div>
              )}
              
              <MicrophoneRecorder
                onRecordingComplete={handleRecordingComplete}
                maxDuration={5}
                autoAnalyze={true}
                showPlayback={false}
              />
              
              <div className="step-controls">
                <button className="btn btn--ghost" onClick={() => setStep(2)}>
                  Назад
                </button>
              </div>
            </div>
          )}

          {step === 4 && (
            <div className="setup-step">
              <div className="step-icon">
                <Save />
              </div>
              <h3>Підтвердження діапазону</h3>
              <p>Перевірте ваш вокальний діапазон:</p>
              
              <div className="range-summary">
                <div className="note-range">
                  <div className="note-item">
                    <span className="note-label">Мінімальна нота:</span>
                    <span className="note-value">{minNote?.noteName}</span>
                    <span className="note-freq">({minNote?.frequency.toFixed(1)} Гц)</span>
                  </div>
                  <div className="note-item">
                    <span className="note-label">Максимальна нота:</span>
                    <span className="note-value">{maxNote?.noteName}</span>
                    <span className="note-freq">({maxNote?.frequency.toFixed(1)} Гц)</span>
                  </div>
                  <div className="note-item">
                    <span className="note-label">Діапазон:</span>
                    <span className="note-value">
                      {Math.round(12 * Math.log2(maxNote?.frequency / minNote?.frequency))} півтонів
                    </span>
                  </div>
                </div>
              </div>
              
              {error && (
                <div className="alert alert--error">
                  {error}
                </div>
              )}
              
              <div className="step-controls">
                <button className="btn btn--ghost" onClick={resetSetup}>
                  <RefreshCw />
                  Спроувати знову
                </button>
                <button 
                  className="btn btn--primary" 
                  onClick={handleSave}
                  disabled={isProcessing}
                >
                  {isProcessing ? (
                    <>
                      <LoadingIndicator size="small" />
                      Зберігаємо...
                    </>
                  ) : (
                    <>
                      <Save />
                      Зберегти діапазон
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default VocalRangeSetup;