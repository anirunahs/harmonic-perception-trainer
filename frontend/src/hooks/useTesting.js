import { useState, useCallback } from "react";
import api from "../api";

export const useTesting = () => {
  const [currentSession, setCurrentSession] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [questionIndex, setQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [sessionCompleted, setSessionCompleted] = useState(false);
  const [results, setResults] = useState(null);

  const createTestSession = useCallback(async (testType, options = {}) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await api.post('/api/testing/create-session/', {
        test_type: testType,
        total_questions: options.totalQuestions || 10,
        intervals: options.intervals || [],
        difficulty: options.difficulty || 'medium'
      });

      const session = response.data;
      setCurrentSession(session);
      setCurrentQuestion(session.questions[0]);
      setQuestionIndex(0);
      setAnswers({});
      setSessionCompleted(false);
      setResults(null);
      
      return session;
    } catch (err) {
      const errorMessage = err.response?.data?.error || 'Помилка створення тесту';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const submitAnswer = useCallback(async (answer, recordedFrequency = null) => {
    if (!currentQuestion) return;

    setIsLoading(true);
    setError(null);

    try {
      const payload = {
        question_id: currentQuestion.id,
      };

      if (answer) payload.answer = answer;
      if (recordedFrequency) payload.recorded_frequency = recordedFrequency;

      const response = await api.post('/api/testing/submit-answer/', payload);
      
      const { is_correct, session_completed } = response.data;
      
      setAnswers(prev => ({
        ...prev,
        [currentQuestion.id]: {
          answer: answer || recordedFrequency,
          is_correct,
          question: currentQuestion
        }
      }));

      if (session_completed) {
        setSessionCompleted(true);
        const sessionResponse = await api.get(`/api/testing/sessions/${currentSession.id}/`);
        setResults(sessionResponse.data);
      } else {
        const nextIndex = questionIndex + 1;
        if (nextIndex < currentSession.questions.length) {
          setQuestionIndex(nextIndex);
          setCurrentQuestion(currentSession.questions[nextIndex]);
        }
      }

      return { is_correct, session_completed };
    } catch (err) {
      const errorMessage = err.response?.data?.error || 'Помилка відправки відповіді';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [currentQuestion, questionIndex, currentSession]);

  const nextQuestion = useCallback(() => {
    if (!currentSession || questionIndex >= currentSession.questions.length - 1) return;
    
    const nextIndex = questionIndex + 1;
    setQuestionIndex(nextIndex);
    setCurrentQuestion(currentSession.questions[nextIndex]);
  }, [currentSession, questionIndex]);

  const previousQuestion = useCallback(() => {
    if (questionIndex <= 0) return;
    
    const prevIndex = questionIndex - 1;
    setQuestionIndex(prevIndex);
    setCurrentQuestion(currentSession.questions[prevIndex]);
  }, [questionIndex, currentSession]);

  const resetTest = useCallback(() => {
    setCurrentSession(null);
    setCurrentQuestion(null);
    setQuestionIndex(0);
    setAnswers({});
    setSessionCompleted(false);
    setResults(null);
    setError(null);
  }, []);

  const getTestProgress = useCallback(() => {
    if (!currentSession) return { current: 0, total: 0, percentage: 0 };
    
    const answeredCount = Object.keys(answers).length;
    const total = currentSession.questions.length;
    const percentage = total > 0 ? (answeredCount / total) * 100 : 0;
    
    return {
      current: answeredCount,
      total,
      percentage: Math.round(percentage)
    };
  }, [currentSession, answers]);

  const getCorrectAnswersCount = useCallback(() => {
    return Object.values(answers).filter(answer => answer.is_correct).length;
  }, [answers]);

  return {
    currentSession,
    currentQuestion,
    questionIndex,
    answers,
    isLoading,
    error,
    sessionCompleted,
    results,
    
    createTestSession,
    submitAnswer,
    nextQuestion,
    previousQuestion,
    resetTest,
    
    getTestProgress,
    getCorrectAnswersCount,
    
    setError
  };
};
