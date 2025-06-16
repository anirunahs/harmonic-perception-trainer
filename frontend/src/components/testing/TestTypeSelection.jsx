import React from "react";
import { BookOpen, Timer, Star, Info } from "lucide-react";

const TestTypeSelection = ({ selectedTestType, onSelectTestType }) => {
  const testTypes = [
    {
      id: "interval_recognition",
      name: "Розпізнавання інтервалів",
      description: "Прослухайте музичний інтервал та оберіть правильну назву",
      icon: "🎵",
      difficulty: "medium",
      estimatedTime: "5-15 хв",
      skills: ["Слух", "Теорія музики", "Розпізнавання"],
      tips: "Почніть з простих інтервалів як кварта та квінта"
    },
    {
      id: "note_reproduction",
      name: "Відтворення нот",
      description: "Прослухайте ноту та відтворіть її голосом",
      icon: "🎤",
      difficulty: "hard",
      estimatedTime: "10-20 хв",
      skills: ["Вокал", "Інтонація", "Музичний слух"],
      tips: "Налаштуйте свій вокальний діапазон для кращих результатів"
    },
  ];

  return (
    <div className="test-type-selection">
      <h2 className="section-title">
        <BookOpen />
        Оберіть тип тесту
      </h2>
      
      <div className="test-types-grid">
        {testTypes.map((testType) => (
          <button
            key={testType.id}
            className={`test-type-card ${
              selectedTestType === testType.id ? 'test-type-card--selected' : ''
            }`}
            onClick={() => onSelectTestType(testType.id)}
          >
            <div className="test-type-card__icon">{testType.icon}</div>
            <h3 className="test-type-card__name">{testType.name}</h3>
            <p className="test-type-card__description">{testType.description}</p>
            
            <div className="test-type-card__meta">
              <div className="test-meta-item">
                <Timer className="meta-icon" />
                <span>{testType.estimatedTime}</span>
              </div>
              <div className="test-meta-item">
                <Star className="meta-icon" />
                <span>{testType.difficulty === 'easy' ? 'Легкий' : testType.difficulty === 'medium' ? 'Середній' : 'Складний'}</span>
              </div>
            </div>

            <div className="test-type-card__skills">
              <span className="skills-label">Навички:</span>
              <div className="skills-list">
                {testType.skills.map(skill => (
                  <span key={skill} className="skill-tag">{skill}</span>
                ))}
              </div>
            </div>

            <div className="test-type-card__tip">
              <Info className="tip-icon" />
              <span>{testType.tips}</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default TestTypeSelection;