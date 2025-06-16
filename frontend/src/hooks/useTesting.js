import { useState, useCallback, useRef, useEffect } from "react";
import api from "../api";

export const useTesting = () => {
  // Основні стани
  const [currentSession, setCurrentSession] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [questionIndex, setQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [sessionCompleted, setSessionCompleted] = useState(false);
  const [results, setResults] = useState(null);

  // Додаткові стани
  const [timeElapsed, setTimeElapsed] = useState(0);
  const [questionStartTime, setQuestionStartTime] = useState(null);
  const [questionTimeLeft, setQuestionTimeLeft] = useState(null);
  const [sessionPaused, setSessionPaused] = useState(false);
  const [confidence, setConfidence] = useState({});
  const [reviewMode, setReviewMode] = useState(false);
  const [hints, setHints] = useState({});
  const [streakCount, setStreakCount] = useState(0);
  const [longestStreak, setLongestStreak] = useState(0);
  
  // Рефи для таймерів
  const sessionTimerRef = useRef(null);
  const questionTimerRef = useRef(null);
  const startTimeRef = useRef(null);

  // Для відстеження часу сесії
  useEffect(() => {
    if (currentSession && !sessionCompleted && !sessionPaused) {
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
  }, [currentSession, sessionCompleted, sessionPaused]);

  // Для таймера питання
  useEffect(() => {
    if (currentQuestion && questionTimeLeft && !sessionPaused) {
      questionTimerRef.current = setInterval(() => {
        setQuestionTimeLeft(prev => {
          if (prev <= 1) {
            handleTimeOut();
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
  }, [currentQuestion, questionTimeLeft, sessionPaused]);

  // Створення тестової сесії
  const createTestSession = useCallback(async (testType, options = {}) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await api.post('/api/testing/create-session/', {
        test_type: testType,
        total_questions: options.totalQuestions || 10,
        intervals: options.intervals || [],
        difficulty: options.difficulty || 'medium',
        time_limit: options.timeLimit,
        enable_hints: options.enableHints || false,
        enable_confidence: options.enableConfidence || false
      });

      const session = response.data;
      setCurrentSession(session);
      setCurrentQuestion(session.questions[0]);
      setQuestionIndex(0);
      setAnswers({});
      setConfidence({});
      setHints({});
      setSessionCompleted(false);
      setResults(null);
      setTimeElapsed(0);
      setStreakCount(0);
      setLongestStreak(0);
      setReviewMode(false);
      setQuestionStartTime(Date.now());
      
      if (options.timeLimit) {
        setQuestionTimeLeft(options.timeLimit);
      }
      
      startTimeRef.current = Date.now();
      
      return session;
    } catch (err) {
      const errorMessage = err.response?.data?.error || 'Помилка створення тесту';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Відправка відповіді
  const submitAnswer = useCallback(async (answer, confidenceLevel = null, recordedFrequency = null) => {
    if (!currentQuestion) return;

    const questionTime = questionStartTime ? (Date.now() - questionStartTime) / 1000 : 0;

    setIsLoading(true);
    setError(null);

    try {
      const payload = {
        question_id: currentQuestion.id,
        response_time: questionTime
      };

      if (answer) payload.answer = answer;
      if (recordedFrequency) payload.recorded_frequency = recordedFrequency;
      if (confidenceLevel) payload.confidence = confidenceLevel;

      const response = await api.post('/api/testing/submit-answer/', payload);
      
      const { is_correct, session_completed } = response.data;
      
      // Оновлення відповідей
      setAnswers(prev => ({
        ...prev,
        [currentQuestion.id]: {
          answer: answer || recordedFrequency,
          is_correct,
          question: currentQuestion,
          response_time: questionTime,
          confidence: confidenceLevel
        }
      }));

      // Оновлення рівня впевненості
      if (confidenceLevel) {
        setConfidence(prev => ({
          ...prev,
          [currentQuestion.id]: confidenceLevel
        }));
      }

      // Оновлення статистики серій
      if (is_correct) {
        setStreakCount(prev => {
          const newStreak = prev + 1;
          setLongestStreak(current => Math.max(current, newStreak));
          return newStreak;
        });
      } else {
        setStreakCount(0);
      }

      if (session_completed) {
        setSessionCompleted(true);
        const sessionResponse = await api.get(`/api/testing/sessions/${currentSession.id}/`);
        setResults({
          ...sessionResponse.data,
          total_time: timeElapsed,
          longest_streak: longestStreak,
          final_streak: streakCount
        });
      } else {
        const nextIndex = questionIndex + 1;
        if (nextIndex < currentSession.questions.length) {
          setQuestionIndex(nextIndex);
          setCurrentQuestion(currentSession.questions[nextIndex]);
          setQuestionStartTime(Date.now());
          
          if (currentSession.time_limit) {
            setQuestionTimeLeft(currentSession.time_limit);
          }
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
  }, [currentQuestion, questionStartTime, questionIndex, currentSession, timeElapsed, longestStreak, streakCount]);

  // Обробка закінчення часу на питання
  const handleTimeOut = useCallback(() => {
    if (currentQuestion && !answers[currentQuestion.id]) {
      submitAnswer(null, null);
    }
  }, [currentQuestion, answers, submitAnswer]);

  // Навігація між питаннями
  const nextQuestion = useCallback(() => {
    if (!currentSession || questionIndex >= currentSession.questions.length - 1) return;
    
    const nextIndex = questionIndex + 1;
    setQuestionIndex(nextIndex);
    setCurrentQuestion(currentSession.questions[nextIndex]);
    setQuestionStartTime(Date.now());
    
    if (currentSession.time_limit) {
      setQuestionTimeLeft(currentSession.time_limit);
    }
  }, [currentSession, questionIndex]);

  const previousQuestion = useCallback(() => {
    if (questionIndex <= 0) return;
    
    const prevIndex = questionIndex - 1;
    setQuestionIndex(prevIndex);
    setCurrentQuestion(currentSession.questions[prevIndex]);
    setQuestionStartTime(Date.now());
    
    if (currentSession.time_limit) {
      setQuestionTimeLeft(currentSession.time_limit);
    }
  }, [questionIndex, currentSession]);

  // Перехід до конкретного питання
  const goToQuestion = useCallback((index) => {
    if (!currentSession || index < 0 || index >= currentSession.questions.length) return;
    
    setQuestionIndex(index);
    setCurrentQuestion(currentSession.questions[index]);
    setQuestionStartTime(Date.now());
    
    if (currentSession.time_limit) {
      setQuestionTimeLeft(currentSession.time_limit);
    }
  }, [currentSession]);

  // Пауза/відновлення сесії
  const togglePause = useCallback(() => {
    setSessionPaused(prev => !prev);
  }, []);

  // Режим перегляду
  const enterReviewMode = useCallback(() => {
    setReviewMode(true);
    setSessionPaused(true);
  }, []);

  const exitReviewMode = useCallback(() => {
    setReviewMode(false);
    setSessionPaused(false);
  }, []);

  // Отримання підказки
  const getHint = useCallback(async (questionId) => {
    if (!questionId || hints[questionId]) return;

    try {
      const response = await api.post('/api/testing/get-hint/', {
        question_id: questionId
      });
      
      setHints(prev => ({
        ...prev,
        [questionId]: response.data.hint
      }));
      
      return response.data.hint;
    } catch (err) {
      console.error('Помилка отримання підказки:', err);
      return null;
    }
  }, [hints]);

  // Скидання тесту
  const resetTest = useCallback(() => {
    if (sessionTimerRef.current) {
      clearInterval(sessionTimerRef.current);
    }
    if (questionTimerRef.current) {
      clearInterval(questionTimerRef.current);
    }

    setCurrentSession(null);
    setCurrentQuestion(null);
    setQuestionIndex(0);
    setAnswers({});
    setConfidence({});
    setHints({});
    setSessionCompleted(false);
    setResults(null);
    setError(null);
    setTimeElapsed(0);
    setQuestionStartTime(null);
    setQuestionTimeLeft(null);
    setSessionPaused(false);
    setReviewMode(false);
    setStreakCount(0);
    setLongestStreak(0);
    
    startTimeRef.current = null;
  }, []);

  // Статистика прогресу
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

  // Кількість правильних відповідей
  const getCorrectAnswersCount = useCallback(() => {
    return Object.values(answers).filter(answer => answer.is_correct).length;
  }, [answers]);

  // Поточна точність
  const getCurrentAccuracy = useCallback(() => {
    const totalAnswered = Object.keys(answers).length;
    if (totalAnswered === 0) return 0;
    
    const correctCount = getCorrectAnswersCount();
    return Math.round((correctCount / totalAnswered) * 100);
  }, [answers, getCorrectAnswersCount]);

  // Статистика за часом
  const getTimeStatistics = useCallback(() => {
    const responseTimes = Object.values(answers)
      .filter(answer => answer.response_time)
      .map(answer => answer.response_time);
    
    if (responseTimes.length === 0) {
      return { average: 0, fastest: 0, slowest: 0 };
    }
    
    const average = responseTimes.reduce((sum, time) => sum + time, 0) / responseTimes.length;
    const fastest = Math.min(...responseTimes);
    const slowest = Math.max(...responseTimes);
    
    return {
      average: Math.round(average * 10) / 10,
      fastest: Math.round(fastest * 10) / 10,
      slowest: Math.round(slowest * 10) / 10
    };
  }, [answers]);

  // Статистика за рівнем впевненості
  const getConfidenceStatistics = useCallback(() => {
    const confidenceEntries = Object.entries(confidence);
    if (confidenceEntries.length === 0) return null;
    
    const stats = { high: 0, medium: 0, low: 0 };
    
    confidenceEntries.forEach(([questionId, level]) => {
      const answer = answers[questionId];
      if (answer && answer.is_correct) {
        stats[level]++;
      }
    });
    
    return stats;
  }, [confidence, answers]);

  // Отримання питань за статусом
  const getQuestionsByStatus = useCallback(() => {
    const correct = [];
    const incorrect = [];
    const unanswered = [];
    
    currentSession?.questions.forEach((question, index) => {
      const answer = answers[question.id];
      if (!answer) {
        unanswered.push({ question, index });
      } else if (answer.is_correct) {
        correct.push({ question, index, answer });
      } else {
        incorrect.push({ question, index, answer });
      }
    });
    
    return { correct, incorrect, unanswered };
  }, [currentSession, answers]);

  // Форматування часу
  const formatTime = useCallback((seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  }, []);

  // Збереження прогресу
  const saveProgress = useCallback(async () => {
    if (!currentSession) return;

    try {
      await api.post('/api/testing/save-progress/', {
        session_id: currentSession.id,
        answers,
        current_question_index: questionIndex,
        time_elapsed: timeElapsed
      });
    } catch (err) {
      console.error('Помилка збереження прогресу:', err);
    }
  }, [currentSession, answers, questionIndex, timeElapsed]);

  // Відновлення збереженого прогресу
  const loadProgress = useCallback(async (sessionId) => {
    try {
      const response = await api.get(`/api/testing/load-progress/${sessionId}/`);
      const progressData = response.data;
      
      setCurrentSession(progressData.session);
      setAnswers(progressData.answers || {});
      setQuestionIndex(progressData.current_question_index || 0);
      setCurrentQuestion(progressData.session.questions[progressData.current_question_index || 0]);
      setTimeElapsed(progressData.time_elapsed || 0);
      
      return progressData;
    } catch (err) {
      console.error('Помилка завантаження прогресу:', err);
      throw err;
    }
  }, []);

  return {
    currentSession,
    currentQuestion,
    questionIndex,
    answers,
    isLoading,
    error,
    sessionCompleted,
    results,
    
    timeElapsed,
    questionTimeLeft,
    sessionPaused,
    reviewMode,
    confidence,
    hints,
    streakCount,
    longestStreak,
    
    createTestSession,
    submitAnswer,
    nextQuestion,
    previousQuestion,
    goToQuestion,
    resetTest,
    
    togglePause,
    enterReviewMode,
    exitReviewMode,
    getHint,
    saveProgress,
    loadProgress,
    
    getTestProgress,
    getCorrectAnswersCount,
    getCurrentAccuracy,
    getTimeStatistics,
    getConfidenceStatistics,
    getQuestionsByStatus,
    
    formatTime,
    
    setError,
    setQuestionTimeLeft,
    setReviewMode
  };
};