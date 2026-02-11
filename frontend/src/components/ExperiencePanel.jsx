/**
 * Experience Panel Component
 * 
 * Displays user's current level, experience points, and progress to next level.
 * Minimalist, elegant design.
 */

import React from "react";
import { Award, TrendingUp } from "lucide-react";

const ExperiencePanel = ({ profile, isLoading }) => {
  if (isLoading || !profile) {
    return (
      <div className="experience-panel experience-panel--loading">
        <div className="experience-panel__skeleton">
          <div className="skeleton-line skeleton-line--short"></div>
          <div className="skeleton-line skeleton-line--long"></div>
        </div>
      </div>
    );
  }

  const { level, experience_points, xp_for_next_level } = profile;
  const currentLevelXP = calculateCurrentLevelXP(experience_points, level);
  // xp_for_next_level з бекенду — це вже скільки XP залишилось до наступного рівня
  const remainingXP = xp_for_next_level != null ? xp_for_next_level : 100;
  // Прогрес у межах поточного рівня: скільки набрано / (набрано + залишилось)
  const xpRequiredThisLevel = currentLevelXP + remainingXP;
  const progressPercentage = xpRequiredThisLevel > 0
    ? Math.min((currentLevelXP / xpRequiredThisLevel) * 100, 100)
    : 0;

  return (
    <div className="experience-panel">
      <div className="experience-panel__content">
        <div className="experience-panel__level">
          <Award className="experience-panel__icon" />
          <span className="experience-panel__level-text">Рівень {level}</span>
        </div>
        
        <div className="experience-panel__progress">
        <div className="experience-panel__xp-info">
          <span className="experience-panel__xp-current">{experience_points}</span>
          <span className="experience-panel__xp-separator">/</span>
          <span className="experience-panel__xp-next">
            {experience_points + remainingXP}
          </span>
        </div>
          
          <div className="experience-panel__progress-bar">
            <div 
              className="experience-panel__progress-fill"
              style={{ width: `${progressPercentage}%` }}
            >
              <div className="experience-panel__progress-shine"></div>
            </div>
          </div>
          
          <div className="experience-panel__xp-remaining">
            <TrendingUp className="experience-panel__trend-icon" />
            <span>{remainingXP} XP до наступного рівня</span>
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * Calculate XP within current level.
 * Based on UserProfile.calculate_level logic:
 * Level 1: 0-100 XP
 * Level 2: 100-250 XP (100 + 150)
 * Level 3: 250-450 XP (250 + 200)
 * etc.
 */
function calculateCurrentLevelXP(totalXP, currentLevel) {
  if (currentLevel === 1) {
    return totalXP;
  }
  
  let levelStartXP = 0;
  let requiredXP = 100;
  
  for (let i = 1; i < currentLevel; i++) {
    levelStartXP += requiredXP;
    requiredXP = 100 + (i * 50); // 100, 150, 200, 250...
  }
  
  return totalXP - levelStartXP;
}

export default ExperiencePanel;
