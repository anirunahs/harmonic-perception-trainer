/**
 * Hook for managing test statistics (streaks, hints).
 */

import { useState, useCallback } from "react";

export const useTestStats = () => {
  const [streakCount, setStreakCount] = useState(0);
  const [longestStreak, setLongestStreak] = useState(0);
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

  const useHint = useCallback((questionId) => {
    setHints(prev => ({
      ...prev,
      [questionId]: true
    }));
  }, []);

  const resetStats = useCallback(() => {
    setStreakCount(0);
    setLongestStreak(0);
    setHints({});
  }, []);

  return {
    streakCount,
    longestStreak,
    hints,
    updateStreak,
    useHint,
    resetStats,
  };
};
