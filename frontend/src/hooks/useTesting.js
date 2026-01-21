/**
 * Refactored useTesting hook that combines smaller hooks.
 * Better organized and easier to maintain.
 */

import { useCallback, useEffect } from "react";
import { useTestSession } from "./useTestSession";
import { useTestProgress } from "./useTestProgress";
import { useTestTimer } from "./useTestTimer";
import { useTestStats } from "./useTestStats";

export const useTesting = () => {
  // Session management
  const {
    currentSession,
    isLoading: sessionLoading,
    error: sessionError,
    createTestSession,
    submitAnswer: submitAnswerToAPI,
    resetSession,
    setError: setSessionError,
  } = useTestSession();

  // Progress management
  const {
    currentQuestion,
    questionIndex,
    answers,
    sessionCompleted,
    results,
    nextQuestion: progressNextQuestion,
    previousQuestion: progressPreviousQuestion,
    goToQuestion,
    updateAnswer,
    completeSession,
    resetProgress,
    getProgress,
  } = useTestProgress(currentSession);

  // Pause state
  const [isPaused, setIsPaused] = useState(false);

  // Timer management
  const {
    timeElapsed,
    questionTimeLeft,
    questionStartTime,
    resetSessionTimer,
    resetQuestionTimer,
    getQuestionTime,
  } = useTestTimer(
    currentSession,
    isPaused,
    currentSession?.time_limit || null
  );

  // Statistics
  const {
    streakCount,
    longestStreak,
    hints,
    updateStreak,
    useHint,
    resetStats,
  } = useTestStats();

  // Initialize question timer when question changes
  useEffect(() => {
    if (currentQuestion) {
      resetQuestionTimer();
    }
  }, [currentQuestion, resetQuestionTimer]);

  // Submit answer with all logic
  const submitAnswer = useCallback(async (answer) => {
    if (!currentQuestion || !currentSession || !answer) return;

    const responseTime = getQuestionTime();

    try {
      const result = await submitAnswerToAPI(
        currentQuestion.id,
        answer,
        responseTime
      );

      // Update answer in progress
      updateAnswer(currentQuestion.id, {
        answer: answer,
        is_correct: result.is_correct,
        question: currentQuestion,
        response_time: responseTime,
      });

      // Update streak
      updateStreak(result.is_correct);

      // Handle session completion
      if (result.session_completed) {
        completeSession({
          ...currentSession,
          total_time: timeElapsed,
          longest_streak: longestStreak,
          final_streak: streakCount + (result.is_correct ? 1 : 0),
        });
      } else {
        // Auto-advance to next question
        progressNextQuestion();
      }

      return result;
    } catch (error) {
      throw error;
    }
  }, [
    currentQuestion,
    currentSession,
    submitAnswerToAPI,
    getQuestionTime,
    updateAnswer,
    updateStreak,
    completeSession,
    timeElapsed,
    longestStreak,
    streakCount,
    progressNextQuestion,
  ]);

  // Create test session with initialization
  const createSession = useCallback(async (testType, options = {}) => {
    try {
      const session = await createTestSession(testType, options);
      
      // Reset all state
      resetProgress();
      resetStats();
      resetSessionTimer();
      
      return session;
    } catch (error) {
      throw error;
    }
  }, [createTestSession, resetProgress, resetStats, resetSessionTimer]);

  // Toggle pause
  const togglePause = useCallback(() => {
    setIsPaused(prev => !prev);
  }, []);

  // Reset everything
  const resetTest = useCallback(() => {
    resetSession();
    resetProgress();
    resetStats();
    setIsPaused(false);
  }, [resetSession, resetProgress, resetStats]);

  // Navigation
  const nextQuestion = useCallback(() => {
    progressNextQuestion();
    resetQuestionTimer();
  }, [progressNextQuestion, resetQuestionTimer]);

  const previousQuestion = useCallback(() => {
    progressPreviousQuestion();
    resetQuestionTimer();
  }, [progressPreviousQuestion, resetQuestionTimer]);

  return {
    // Session
    currentSession,
    isLoading: sessionLoading,
    error: sessionError,
    createTestSession: createSession,
    resetTest,
    setError: setSessionError,

    // Progress
    currentQuestion,
    questionIndex,
    answers,
    sessionCompleted,
    results,
    nextQuestion,
    previousQuestion,
    goToQuestion,
    getTestProgress: getProgress,

    // Timer
    timeElapsed,
    questionTimeLeft,
    isPaused,
    togglePause,
    getQuestionTime,

    // Statistics
    streakCount,
    longestStreak,
    hints,
    useHint,

    // Answer submission
    submitAnswer,

    // Helpers
    getCorrectAnswersCount: () => {
      return Object.values(answers).filter(a => a.is_correct).length;
    },
    getCurrentAccuracy: () => {
      const totalAnswered = Object.keys(answers).length;
      if (totalAnswered === 0) return 0;
      const correct = Object.values(answers).filter(a => a.is_correct).length;
      return Math.round((correct / totalAnswered) * 100);
    },
  };
};
