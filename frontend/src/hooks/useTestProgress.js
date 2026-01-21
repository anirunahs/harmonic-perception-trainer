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

  // Initialize question when session changes (but not if already completed)
  useEffect(() => {
    if (session && session.questions && session.questions.length > 0 && !sessionCompleted) {
      setCurrentQuestion(session.questions[0]);
      setQuestionIndex(0);
      setSessionCompleted(false);
      setResults(null);
    }
  }, [session, sessionCompleted]);

  // Sync currentQuestion with questionIndex (but not if session is completed)
  useEffect(() => {
    if (session && session.questions && session.questions.length > 0 && !sessionCompleted) {
      if (questionIndex >= 0 && questionIndex < session.questions.length) {
        setCurrentQuestion(session.questions[questionIndex]);
      }
    }
  }, [session, questionIndex, sessionCompleted]);

  const nextQuestion = useCallback(() => {
    if (!session || !session.questions) return;
    
    setQuestionIndex(prevIndex => {
      const nextIndex = prevIndex + 1;
      if (nextIndex < session.questions.length) {
        setCurrentQuestion(session.questions[nextIndex]);
        return nextIndex;
      }
      return prevIndex; // Don't change if at last question
    });
  }, [session]);

  const previousQuestion = useCallback(() => {
    if (!session || !session.questions) return;
    
    setQuestionIndex(prevIndex => {
      if (prevIndex > 0) {
        const newIndex = prevIndex - 1;
        setCurrentQuestion(session.questions[newIndex]);
        return newIndex;
      }
      return prevIndex; // Don't change if at first question
    });
  }, [session]);

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
    // Don't change currentQuestion when session is completed
    // Keep showing the last question or results
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
