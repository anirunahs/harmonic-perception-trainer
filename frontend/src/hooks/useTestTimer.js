/**
 * Hook for managing test timers (session timer and question timer).
 */

import { useState, useEffect, useRef, useCallback } from "react";

export const useTestTimer = (session, isPaused = false, questionTimeLimit = null) => {
  const [timeElapsed, setTimeElapsed] = useState(0);
  const [questionTimeLeft, setQuestionTimeLeft] = useState(questionTimeLimit);
  const [questionStartTime, setQuestionStartTime] = useState(null);
  
  const sessionTimerRef = useRef(null);
  const questionTimerRef = useRef(null);
  const startTimeRef = useRef(null);

  // Session timer
  useEffect(() => {
    if (session && !isPaused) {
      if (!startTimeRef.current) {
        startTimeRef.current = Date.now();
      }

      sessionTimerRef.current = setInterval(() => {
        setTimeElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
      }, 1000);

      return () => {
        if (sessionTimerRef.current) {
          clearInterval(sessionTimerRef.current);
        }
      };
    } else {
      if (sessionTimerRef.current) {
        clearInterval(sessionTimerRef.current);
      }
    }
  }, [session, isPaused]);

  // Question timer
  useEffect(() => {
    if (questionTimeLimit && !isPaused) {
      setQuestionTimeLeft(questionTimeLimit);
      setQuestionStartTime(Date.now());

      questionTimerRef.current = setInterval(() => {
        setQuestionTimeLeft(prev => {
          if (prev <= 1) {
            return 0;
          }
          return prev - 1;
        });
      }, 1000);

      return () => {
        if (questionTimerRef.current) {
          clearInterval(questionTimerRef.current);
        }
      };
    } else {
      if (questionTimerRef.current) {
        clearInterval(questionTimerRef.current);
      }
    }
  }, [questionTimeLimit, isPaused]);

  const resetSessionTimer = useCallback(() => {
    startTimeRef.current = Date.now();
    setTimeElapsed(0);
  }, []);

  const resetQuestionTimer = useCallback(() => {
    setQuestionStartTime(Date.now());
    if (questionTimeLimit) {
      setQuestionTimeLeft(questionTimeLimit);
    }
  }, [questionTimeLimit]);

  const getQuestionTime = useCallback(() => {
    if (!questionStartTime) return 0;
    return (Date.now() - questionStartTime) / 1000;
  }, [questionStartTime]);

  return {
    timeElapsed,
    questionTimeLeft,
    questionStartTime,
    resetSessionTimer,
    resetQuestionTimer,
    getQuestionTime,
  };
};
