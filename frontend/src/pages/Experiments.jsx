import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import Header from "../components/Header";
import api from "../api";
import { FlaskConical, ArrowRight } from "lucide-react";

function Experiments() {
  const [list, setList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    api
      .get("/api/experiments/")
      .then((res) => setList(res.data.experiments || []))
      .catch((err) => setError(err.response?.data?.detail || "Помилка завантаження"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <>
      <Header />
      <div className="experiments-page">
        <div className="experiments-page__container">
          <header className="experiments-page__header">
            <h1 className="experiments-page__title">Експерименти</h1>
          </header>

          {loading && <p className="experiments-page__loading">Завантаження...</p>}
          {error && <div className="alert alert--error">{error}</div>}

          {!loading && !error && (
            <div className="experiments-grid">
              {list.map((exp) => (
                <Link key={exp.id} to={exp.path} className="experiment-card">
                  <div className="experiment-card__icon">
                    <FlaskConical aria-hidden />
                  </div>
                  <h2 className="experiment-card__title">{exp.name_uk}</h2>
                  <p className="experiment-card__description">{exp.description_uk}</p>
                  <span className="experiment-card__link">
                    Відкрити <ArrowRight />
                  </span>
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default Experiments;
