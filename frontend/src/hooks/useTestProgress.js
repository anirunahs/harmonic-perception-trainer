/**
 * Hook for managing test progress and navigation.
 */

import { useState, useCallback, useEffect, useRef } from "react";

export const useTestProgress = (session) => {
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [questionIndex, setQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [sessionCompleted, setSessionCompleted] = useState(false);
  const [results, setResults] = useState(null);

  // Initialize question when session changes
  useEffect(() => {
    if (session && session.questions && session.questions.length > 0) {
      setCurrentQuestion(session.questions[0]);
      setQuestionIndex(0);
      setSessionCompleted(false);
      setResults(null);
    }
  }, [session]);

  const nextQuestion = useCallback(() => {
    if (!session || !session.questions) return;
    
    const nextIndex = questionIndex + 1;
    if (nextIndex < session.questions.length) {
      setQuestionIndex(nextIndex);
      setCurrentQuestion(session.questions[nextIndex]);
    }
  }, [session, questionIndex]);

  const previousQuestion = useCallback(() => {
    if (questionIndex > 0) {
      const prevIndex = questionIndex - 1;
      setQuestionIndex(prevIndex);
      setCurrentQuestion(session?.questions[prevIndex]);
    }
  }, [session, questionIndex]);

  const goToQuestion = useCallback((index) => {
    if (!session || !session.questions) return;
    
    if (index >= 0 && index < session.questions.length) {
      setQuestionIndex(index);
      setCurrentQuestion(session.questions[index]);
    }
  }, [session]);

  const updateAnswer = useCallback((questionId, answerData) => {
    setAnswers(prev => ({
      ...prev,
      [questionId]: answerData
    }));
  }, []);

  const completeSession = useCallback((sessionData) => {
    setSessionCompleted(true);
    setResults(sessionData);
  }, []);

  const resetProgress = useCallback(() => {
    setCurrentQuestion(null);
    setQuestionIndex(0);
    setAnswers({});
    setSessionCompleted(false);
    setResults(null);
  }, []);

  const getProgress = useCallback(() => {
    if (!session) return { current: 0, total: 0, percentage: 0 };
    
    return {
      current: questionIndex + 1,
      total: session.total_questions,
      percentage: ((questionIndex + 1) / session.total_questions) * 100
    };
  }, [session, questionIndex]);

  return {
    currentQuestion,
    questionIndex,
    answers,
    sessionCompleted,
    results,
    nextQuestion,
    previousQuestion,
    goToQuestion,
    updateAnswer,
    completeSession,
    resetProgress,
    getProgress,
  };
};
