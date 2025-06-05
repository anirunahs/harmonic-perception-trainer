import { useRef, useEffect, useState } from "react";
import api from "../api";

export const useAudioPlayer = () => {
  const [currentPlaying, setCurrentPlaying] = useState(null);
  const [loadingAudio, setLoadingAudio] = useState(null);
  const [audioError, setAudioError] = useState(null);
  const audioRefs = useRef({});

  useEffect(() => {
    return () => {
      Object.values(audioRefs.current).forEach(audio => {
        if (audio) {
          audio.pause();
          audio.src = '';
        }
      });
    };
  }, []);

  const createAudioElement = (playId) => {
    const audio = new Audio();
    audioRefs.current[playId] = audio;
    
    audio.preload = 'metadata';
    
    audio.onloadstart = () => {
      console.log(`Loading started for ${playId}`);
    };
    
    audio.oncanplay = () => {
      console.log(`Can play ${playId}`);
      setLoadingAudio(null);
    };
    
    audio.onended = () => {
      console.log(`Ended ${playId}`);
      setCurrentPlaying(null);
      setLoadingAudio(null);
    };
    
    audio.onerror = (e) => {
      if (!audioRefs.current[playId]) {
        return;
      }

      console.error("Audio error:", e);
      
      if (!audio.src || audio.src === '' || audio.src === window.location.href) {
        return;
      }
      
      let errorMessage = "Помилка відтворення аудіо";
      if (audio.error) {
        switch (audio.error.code) {
          case audio.error.MEDIA_ERR_ABORTED:
            errorMessage = "Відтворення було перервано";
            break;
          case audio.error.MEDIA_ERR_NETWORK:
            errorMessage = "Помилка мережі при завантаженні аудіо";
            break;
          case audio.error.MEDIA_ERR_DECODE:
            errorMessage = "Помилка декодування аудіо";
            break;
          case audio.error.MEDIA_ERR_SRC_NOT_SUPPORTED:
            errorMessage = "Формат аудіо не підтримується";
            break;
          default:
            errorMessage = `Невідома помилка аудіо (код: ${audio.error.code})`;
        }
      }
      
      setAudioError(errorMessage);
      setCurrentPlaying(null);
      setLoadingAudio(null);
    };

    return audio;
  };

  const playAudio = async (intervalId, playType, generatedIntervals) => {
    const playId = `${intervalId}_${playType}`;
    
    if (currentPlaying && audioRefs.current[currentPlaying]) {
      audioRefs.current[currentPlaying].pause();
      audioRefs.current[currentPlaying].currentTime = 0;
    }

    if (currentPlaying === playId) {
      setCurrentPlaying(null);
      return;
    }

    try {
      setAudioError(null);
      setLoadingAudio(playId);
      
      const interval = generatedIntervals.find(int => int.id === intervalId);
      if (!interval) {
        throw new Error("Інтервал не знайдено");
      }

      if (!audioRefs.current[playId]) {
        createAudioElement(playId);
      }

      const audio = audioRefs.current[playId];
      const audioUrl = playType === 'harmonic' ? interval.harmonic_url : interval.melodic_url;
      
      try {
        const response = await api.get(audioUrl, {
          responseType: 'blob'
        });
        
        const audioBlob = new Blob([response.data], { type: 'audio/wav' });
        const blobUrl = URL.createObjectURL(audioBlob);
        
        audio.src = blobUrl;
        
        const originalOnended = audio.onended;
        audio.onended = () => {
          URL.revokeObjectURL(blobUrl);
          if (originalOnended) originalOnended();
        };
        
      } catch (error) {
        console.error("Помилка завантаження аудіо:", error);
        throw new Error("Не вдалося завантажити аудіофайл");
      }
      
      audio.load();
      setCurrentPlaying(playId);
      
      await new Promise((resolve, reject) => {
        const onCanPlay = () => {
          audio.removeEventListener('canplay', onCanPlay);
          audio.removeEventListener('error', onError);
          resolve();
        };
        
        const onError = (e) => {
          audio.removeEventListener('canplay', onCanPlay);
          audio.removeEventListener('error', onError);
          reject(e);
        };
        
        audio.addEventListener('canplay', onCanPlay);
        audio.addEventListener('error', onError);
        
        setTimeout(() => {
          audio.removeEventListener('canplay', onCanPlay);
          audio.removeEventListener('error', onError);
          reject(new Error('Таймаут завантаження аудіо'));
        }, 10000);
      });
      
      await audio.play();
      setLoadingAudio(null);
      
    } catch (error) {
      console.error("Помилка відтворення:", error);
      setAudioError(`Помилка відтворення: ${error.message}`);
      setCurrentPlaying(null);
      setLoadingAudio(null);
    }
  };

  const stopAudio = () => {
    if (currentPlaying && audioRefs.current[currentPlaying]) {
      const audio = audioRefs.current[currentPlaying];

      const originalOnError = audio.onerror;
      audio.onerror = null;
      audio.pause();
      audio.currentTime = 0;
      
      setTimeout(() => {
        if (audio) {
          audio.onerror = originalOnError;
        }
      }, 100);
      
      setCurrentPlaying(null);
    }
    setLoadingAudio(null);
  };

  const clearAllAudio = () => {
    setCurrentPlaying(null);
    setAudioError(null);
    setLoadingAudio(null);

    Object.values(audioRefs.current).forEach(audio => {
      if (audio) {
        audio.onended = null;
        audio.onerror = null;
        audio.oncanplay = null;
        audio.onloadstart = null;

        audio.pause();
        audio.currentTime = 0;
        audio.src = '';
        audio.load();
      }
    });

    audioRefs.current = {};
  };

  return {
    currentPlaying,
    loadingAudio,
    audioError,
    setAudioError,
    playAudio,
    stopAudio,
    clearAllAudio
  };
};