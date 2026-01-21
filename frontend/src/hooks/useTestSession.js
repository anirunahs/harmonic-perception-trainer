/**
 * Hook for managing test session state and API calls.
 * Separated from useTesting for better organization.
 */

import { useState, useCallback } from "react";
import api from "../api";

export const useTestSession = () => {
  const [currentSession, setCurrentSession] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const createTestSession = useCallback(async (testType, options = {}) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await api.post('/api/testing/create-session/', {
        test_type: testType,
        total_questions: options.totalQuestions || 10,
        intervals: options.intervals || [],
        instrument: options.instrument || 'piano',
      });

      const session = response.data;
      setCurrentSession(session);
      
      return session;
    } catch (err) {
      const errorMessage = err.response?.data?.error || 'Помилка створення тесту';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const submitAnswer = useCallback(async (questionId, answer, responseTime = null) => {
    setIsLoading(true);
    setError(null);

    try {
      const payload = {
        question_id: questionId,
        answer: answer,
      };

      if (responseTime !== null) payload.response_time = responseTime;

      const response = await api.post('/api/testing/submit-answer/', payload);
      
      // Don't refresh session here - let the parent component handle completion
      // This prevents resetting currentQuestion when session completes
      
      return response.data;
    } catch (err) {
      const errorMessage = err.response?.data?.error || 'Помилка відправки відповіді';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [currentSession]);

  const resetSession = useCallback(() => {
    setCurrentSession(null);
    setError(null);
  }, []);

  return {
    currentSession,
    isLoading,
    error,
    createTestSession,
    submitAnswer,
    resetSession,
    setError,
  };
};
