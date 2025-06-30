import React from "react";
import { Link } from "react-router-dom";
import { Mic, Ear, BookOpenCheck, ArrowRight, Music, Star, Target, Brain, Play, Users, Award } from "lucide-react";
import Header from "../components/Header";

function Home() {
  const modules = [
    {
      id: "recognizer",
      title: "Розпізнавання інтервалів",
      description: "Запишіть звук та отримайте AI-аналіз музичних інтервалів з високою точністю",
      icon: Mic,
      color: "primary",
      gradient: "from-blue-500 to-blue-600",
      features: [
        "Розпізнавання 12 типів інтервалів",
        "Аналіз цілісного відтворення", 
        "Топ 3 ймовірності з точністю 92%",
        "Швидкий AI-аналіз"
      ],
      path: "/recognizer",
      status: "ready"
    },
    {
      id: "training",
      title: "Тренування слуху",
      description: "Розвивайте музичний слух через прослуховування згенерованих інтервалів",
      icon: Ear,
      color: "success",
      gradient: "from-green-500 to-green-600",
      features: [
        "Генерація інтервалів у реальному часі",
        "Вибір базової ноти та типів інтервалів",
        "Гармонічне та мелодичне відтворення",
        "Інтерактивна клавіатура фортепіано"
      ],
      path: "/training",
      status: "ready"
    },
    {
      id: "testing",
      title: "Тестування знань",
      description: "Перевірте свої навички розпізнавання інтервалів через структуровані тести",
      icon: BookOpenCheck,
      color: "warning",
      gradient: "from-orange-500 to-orange-600",
      features: [
        "Налаштовувані тести різної складності",
        "Детальна аналітика результатів",
        "Перевірка здібностей"
      ],
      path: "/testing",
      status: "ready"
    }
  ];

  const stats = [
    { label: "Типів інтервалів", value: "12", icon: Music },
    { label: "Точність розпізнавання", value: "92%", icon: Target },
    { label: "Активних користувачів", value: "50+", icon: Users }
  ];

  const getColorClasses = (color) => {
    switch (color) {
      case "primary":
        return "border-blue-200 hover:border-blue-300 bg-gradient-to-br from-blue-50 to-indigo-50";
      case "success": 
        return "border-green-200 hover:border-green-300 bg-gradient-to-br from-green-50 to-emerald-50";
      case "warning":
        return "border-orange-200 hover:border-orange-300 bg-gradient-to-br from-orange-50 to-amber-50";
      default:
        return "border-gray-200 hover:border-gray-300 bg-gradient-to-br from-gray-50 to-slate-50";
    }
  };

  const getIconColorClasses = (color) => {
    switch (color) {
      case "primary":
        return "text-blue-600 bg-blue-100";
      case "success":
        return "text-green-600 bg-green-100";  
      case "warning":
        return "text-orange-600 bg-orange-100";
      default:
        return "text-gray-600 bg-gray-100";
    }
  };

  return (
    <>
      <Header />
      <div className="home-page">
        <section className="hero-section">
          <div className="hero-content">
            <h1 className="hero-title">
              Розвивайте свій <span className="hero-title--accent">музичний слух</span> з допомогою AI
            </h1>
            
            <p className="hero-description">
              HPT (Harmonic Perception Trainer) - це комплексна платформа для навчання та тестування 
              музичних навичок з використанням штучного інтелекту для точного розпізнавання інтервалів.
            </p>

            <div className="hero-actions">
              <Link to="/recognizer" className="btn btn--primary btn--lg hero-btn">
                <Play />
                <span>Почати зараз</span>
              </Link>
              <Link to="/training" className="btn btn--ghost btn--lg hero-btn">
                <Brain />
                <span>Тренування</span>
              </Link>
            </div>
          </div>
        </section>

        <section className="stats-section">
          <div className="stats-grid">
            {stats.map((stat, index) => (
              <div key={index} className="stat-card">
                <div className="stat-card__icon">
                  <stat.icon />
                </div>
                <div className="stat-card__content">
                  <div className="stat-card__value">{stat.value}</div>
                  <div className="stat-card__label">{stat.label}</div>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="modules-section">
          <div className="section-header">
            <h2 className="section-title">Функціональні модулі</h2>
            <p className="section-description">
              Три потужних інструменти для повноцінного навчання музичних інтервалів
            </p>
          </div>

          <div className="modules-grid">
            {modules.map((module) => {
              const IconComponent = module.icon;
              return (
                <div key={module.id} className={`module-card ${getColorClasses(module.color)}`}>
                  <div className="module-card__header">
                    <div className={`module-card__icon ${getIconColorClasses(module.color)}`}>
                      <IconComponent />
                    </div>
                    <div className="module-card__status">
                      <div className="status-indicator status-indicator--ready"></div>
                      <span>Готово до використання</span>
                    </div>
                  </div>

                  <div className="module-card__content">
                    <h3 className="module-card__title">{module.title}</h3>
                    <p className="module-card__description">{module.description}</p>

                    <div className="module-card__features">
                      <h4 className="features-title">Основні можливості:</h4>
                      <ul className="features-list">
                        {module.features.map((feature, index) => (
                          <li key={index} className="features-item">
                            {feature}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  <div className="module-card__footer">
                    <Link to={module.path} className={`module-link bg-gradient-to-r ${module.gradient}`}>
                      <span>Відкрити модуль</span>
                      <ArrowRight />
                    </Link>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="cta-section">
          <div className="cta-content">
            <Award className="cta-icon" />
            <h2 className="cta-title">Готові розпочати навчання?</h2>
            <p className="cta-description">
              Оберіть діяльність
            </p>
            <div className="cta-actions">
              <Link to="/recognizer" className="btn btn--primary btn--lg">
                <Mic />
                <span>Розпізнати інтервал</span>
              </Link>
              <Link to="/training" className="btn btn--ghost btn--lg">
                <Ear />
                <span>Почати тренування</span>
              </Link>
            </div>
          </div>
        </section>
      </div>
    </>
  );
}

export default Home;