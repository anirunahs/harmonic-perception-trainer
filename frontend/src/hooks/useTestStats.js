/**
 * Hook for managing test statistics (streaks, confidence, hints).
 */

import { useState, useCallback } from "react";

export const useTestStats = () => {
  const [streakCount, setStreakCount] = useState(0);
  const [longestStreak, setLongestStreak] = useState(0);
  const [confidence, setConfidence] = useState({});
  const [hints, setHints] = useState({});

  const updateStreak = useCallback((isCorrect) => {
    if (isCorrect) {
      setStreakCount(prev => {
        const newStreak = prev + 1;
        setLongestStreak(current => Math.max(current, newStreak));
        return newStreak;
      });
    } else {
      setStreakCount(0);
    }
  }, []);

  const setQuestionConfidence = useCallback((questionId, confidenceLevel) => {
    setConfidence(prev => ({
      ...prev,
      [questionId]: confidenceLevel
    }));
  }, []);

  const useHint = useCallback((questionId) => {
    setHints(prev => ({
      ...prev,
      [questionId]: true
    }));
  }, []);

  const resetStats = useCallback(() => {
    setStreakCount(0);
    setLongestStreak(0);
    setConfidence({});
    setHints({});
  }, []);

  return {
    streakCount,
    longestStreak,
    confidence,
    hints,
    updateStreak,
    setQuestionConfidence,
    useHint,
    resetStats,
  };
};
