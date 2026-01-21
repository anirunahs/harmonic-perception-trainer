/**
 * Hook for managing user profile and experience data.
 */

import { useState, useEffect, useCallback } from "react";
import api from "../api";

export const useUserProfile = () => {
  const [profile, setProfile] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchProfile = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await api.get('/api/user/profile/');
      setProfile(response.data);
      return response.data;
    } catch (err) {
      const errorMessage = err.response?.data?.error || 'Помилка завантаження профілю';
      setError(errorMessage);
      console.error('Error fetching profile:', err);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  const refreshProfile = useCallback(() => {
    return fetchProfile();
  }, [fetchProfile]);

  return {
    profile,
    isLoading,
    error,
    refreshProfile,
  };
};
