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
        difficulty: options.difficulty || 'medium',
        instrument: options.instrument || 'piano',
        time_limit: options.timeLimit || null,
        enable_hints: options.enableHints || false,
        auto_next: options.autoNext || false,
        random_order: options.randomOrder || false,
        include_reference_note: options.includeReferenceNote !== false,
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

  const submitAnswer = useCallback(async (questionId, answer, recordedFrequency = null, confidence = null, responseTime = null) => {
    setIsLoading(true);
    setError(null);

    try {
      const payload = {
        question_id: questionId,
      };

      if (answer) payload.answer = answer;
      if (recordedFrequency) payload.recorded_frequency = recordedFrequency;
      if (confidence !== null) payload.confidence = confidence;
      if (responseTime !== null) payload.response_time = responseTime;

      const response = await api.post('/api/testing/submit-answer/', payload);
      
      // Refresh session if completed
      if (response.data.session_completed && currentSession) {
        const sessionResponse = await api.get(`/api/testing/sessions/${currentSession.id}/`);
        setCurrentSession(sessionResponse.data);
      }
      
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
