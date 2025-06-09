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

  return {
    currentSession,
    currentQuestion,
    questionIndex,
    answers,
    isLoading,
    error,
    sessionCompleted,
    results,

    setError,
  };
};
