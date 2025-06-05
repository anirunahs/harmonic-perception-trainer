import React from "react";

const PlayingIndicator = () => {
  return (
    <div className="playing-indicator">
      <div className="playing-indicator__bars">
        <div className="playing-indicator__bar"></div>
        <div className="playing-indicator__bar"></div>
        <div className="playing-indicator__bar"></div>
      </div>
      <span>Відтворюється...</span>
    </div>
  );
};

export default PlayingIndicator;
