/**
 * User Profile Context
 * 
 * Provides global user profile state that can be shared across components.
 * Ensures experience panel updates immediately after test completion.
 */

import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import api from "../api";

const UserProfileContext = createContext(null);

export const useUserProfileContext = () => {
  const context = useContext(UserProfileContext);
  if (!context) {
    throw new Error("useUserProfileContext must be used within UserProfileProvider");
  }
  return context;
};

export const UserProfileProvider = ({ children }) => {
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

  const refreshProfile = useCallback(async (retries = 2) => {
    let lastError = null;
    for (let i = 0; i <= retries; i++) {
      try {
        const updatedProfile = await fetchProfile();
        return updatedProfile;
      } catch (err) {
        lastError = err;
        if (i < retries) {
          // Wait before retry (exponential backoff)
          await new Promise(resolve => setTimeout(resolve, 300 * (i + 1)));
        }
      }
    }
    console.error('Failed to refresh profile after retries:', lastError);
    return null;
  }, [fetchProfile]);

  const value = {
    profile,
    isLoading,
    error,
    refreshProfile,
  };

  return (
    <UserProfileContext.Provider value={value}>
      {children}
    </UserProfileContext.Provider>
  );
};
