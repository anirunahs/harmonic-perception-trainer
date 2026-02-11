import React from "react";
import { BookOpen, Timer, Star, Info, Music2, Mic } from "lucide-react";

const TestTypeSelection = ({ selectedTestType, onSelectTestType }) => {
  const testTypes = [
    {
      id: "interval_recognition",
      name: "Розпізнавання інтервалів",
      description: "Прослухайте музичний інтервал та оберіть правильну назву",
      Icon: Music2,
      difficulty: "medium",
      estimatedTime: "5-15 хв",
      skills: ["Слух", "Теорія музики", "Розпізнавання"],
      tips: "Почніть з простих інтервалів як кварта та квінта"
    },
    {
      id: "note_reproduction",
      name: "Відтворення нот",
      description: "Прослухайте ноту та відтворіть її голосом",
      Icon: Mic,
      difficulty: "hard",
      estimatedTime: "10-20 хв",
      skills: ["Вокал", "Інтонація", "Музичний слух"],
      tips: "Налаштуйте свій вокальний діапазон для кращих результатів"
    },
  ];

  return (
    <div className="test-type-selection">
      <div className="test-type-selection__header">
        <h2 className="test-type-selection__title">
          <BookOpen className="test-type-selection__title-icon" aria-hidden />
          <span>Оберіть тип тесту</span>
        </h2>
        <p className="test-type-selection__subtitle">Виберіть один із форматів тесту нижче</p>
      </div>

      <div className="test-types-grid">
        {testTypes.map((testType) => {
          const IconComponent = testType.Icon;
          return (
          <button
            key={testType.id}
            type="button"
            className={`test-type-card ${
              selectedTestType === testType.id ? 'test-type-card--selected' : ''
            }`}
            onClick={() => onSelectTestType(testType.id)}
          >
            <div className="test-type-card__header">
              <div className="test-type-card__icon">
                <IconComponent aria-hidden />
              </div>
              <h3 className="test-type-card__name">{testType.name}</h3>
            </div>
            <div className="test-type-card__body">
              <p className="test-type-card__description">{testType.description}</p>

              <div className="test-type-card__meta">
                <div className="test-meta-item">
                  <Timer className="meta-icon" aria-hidden />
                  <span>{testType.estimatedTime}</span>
                </div>
                <div className="test-meta-item">
                  <Star className="meta-icon" aria-hidden />
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
                <Info className="tip-icon" aria-hidden />
                <span>{testType.tips}</span>
              </div>
            </div>
          </button>
          );
        })}
      </div>
    </div>
  );
};

export default TestTypeSelection;