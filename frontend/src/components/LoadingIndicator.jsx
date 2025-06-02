import React from 'react';

const LoadingIndicator = ({ 
  size = 'medium', 
  type = 'spinner', 
  text = null, 
  className = '' 
}) => {
  const getSizeClass = () => {
    switch (size) {
      case 'small': return 'loader--small';
      case 'large': return 'loader--large';
      default: return '';
    }
  };

  const renderSpinner = () => (
    <div className={`loader ${getSizeClass()}`} />
  );

  const renderDots = () => (
    <div className="dots-loader">
      <div className="dots-loader__dot"></div>
      <div className="dots-loader__dot"></div>
      <div className="dots-loader__dot"></div>
    </div>
  );

  const renderLoader = () => {
    switch (type) {
      case 'dots': return renderDots();
      default: return renderSpinner();
    }
  };

  return (
    <div className={`loading-container ${className}`}>
      {renderLoader()}
      {text && <div className="loading-text">{text}</div>}
    </div>
  );
};

export default LoadingIndicator