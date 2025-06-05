import React from "react";
import { VolumeX } from "lucide-react";

const ErrorMessage = ({ error, onClose }) => {
  if (!error) return null;

  return (
    <div className="status-card status-card--error">
      <div className="status-card__icon">
        <VolumeX />
      </div>
      <div className="status-card__content">
        <h3 className="status-card__title">Помилка відтворення</h3>
        <p className="status-card__description">{error}</p>
        <button 
          className="btn btn--ghost btn--sm"
          onClick={onClose}
        >
          Закрити
        </button>
      </div>
    </div>
  );
};

export default ErrorMessage;