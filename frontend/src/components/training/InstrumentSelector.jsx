import React from "react";
import { Music, Piano, Guitar } from "lucide-react";

const INSTRUMENTS = [
  { id: "piano", name: "Фортепіано", icon: Piano },
  { id: "guitar", name: "Гітара", icon: Guitar },
];

const InstrumentSelector = ({ selectedInstrument, onInstrumentSelect }) => {
  return (
    <div className="setting-group">
      <label className="setting-group__label">
        <Music size={18} />
        Оберіть інструмент
      </label>
      <div className="instrument-selector">
        {INSTRUMENTS.map((instrument) => (
          <button
            key={instrument.id}
            className={`instrument-btn ${
              selectedInstrument === instrument.id ? "instrument-btn--active" : ""
            }`}
            onClick={() => onInstrumentSelect(instrument.id)}
            title={instrument.name}
          >
            <instrument.icon className="instrument-btn__icon" size={24} />
            <span className="instrument-btn__name">{instrument.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default InstrumentSelector;
