import React from "react";
import { Link } from "react-router-dom";
import { Home, ArrowLeft, Music, SearchX, RefreshCw } from "lucide-react";
import Header from "../components/Header";

function NotFound() {
  const suggestions = [
    {
      title: "Розпізнавання інтервалів",
      description: "Аналізуйте музичні інтервали за допомогою AI",
      path: "/recognizer",
      icon: SearchX
    },
    {
      title: "Тренування слуху",
      description: "Розвивайте навички через прослуховування",
      path: "/training", 
      icon: Music
    },
    {
      title: "Тестування знань",
      description: "Перевірте свої музичні навички",
      path: "/testing",
      icon: RefreshCw
    }
  ];

  return (
    <>
      <Header />
      <div className="not-found-page">
        <div className="not-found-container">
          <div className="error-visual">
            <div className="error-number">4</div>
            <div className="error-icon">
              <Music />
            </div>
            <div className="error-number">4</div>
          </div>

          <div className="error-content">
            <h1 className="error-title">Сторінка не знайдена</h1>
            <p className="error-description">
              На жаль, ми не змогли знайти сторінку, яку ви шукаєте. 
              Можливо, вона була переміщена, видалена або ви ввели неправильну адресу.
            </p>

            <div className="error-actions">
              <Link to="/" className="btn btn--primary btn--lg">
                <Home />
                <span>На головну</span>
              </Link>
              <button 
                onClick={() => window.history.back()} 
                className="btn btn--ghost btn--lg"
              >
                <ArrowLeft />
                <span>Назад</span>
              </button>
            </div>
          </div>

          <div className="suggestions-section">
            <h2 className="suggestions-title">Можливо, вас зацікавить:</h2>
            <div className="suggestions-grid">
              {suggestions.map((suggestion, index) => {
                const IconComponent = suggestion.icon;
                return (
                  <Link 
                    key={index}
                    to={suggestion.path}
                    className="suggestion-card"
                  >
                    <div className="suggestion-card__icon">
                      <IconComponent />
                    </div>
                    <div className="suggestion-card__content">
                      <h3 className="suggestion-card__title">{suggestion.title}</h3>
                      <p className="suggestion-card__description">{suggestion.description}</p>
                    </div>
                    <div className="suggestion-card__arrow">
                      <ArrowLeft />
                    </div>
                  </Link>
                );
              })}
            </div>
          </div>

          <div className="help-section">
            <div className="help-card">
              <h3 className="help-title">Потрібна допомога?</h3>
              <p className="help-description">
                Якщо ви вважаєте, що це помилка, або потребуєте допомоги з навігацією по сайту, 
                зв'яжіться з нами.
              </p>
              <div className="help-info">
                <div className="help-item">
                  <span className="help-label">Код помилки:</span>
                  <span className="help-value">404 - Сторінка не знайдена</span>
                </div>
                <div className="help-item">
                  <span className="help-label">Час:</span>
                  <span className="help-value">{new Date().toLocaleString('uk-UA')}</span>
                </div>
                <div className="help-item">
                  <span className="help-label">URL:</span>
                  <span className="help-value">{window.location.pathname}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default NotFound;