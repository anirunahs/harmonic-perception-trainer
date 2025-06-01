import { Link, useLocation } from "react-router-dom";
import { Music, LogOut, Home, Mic, Ear, BookOpenCheck } from "lucide-react";

function Header() {
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  const navItems = [
    { 
      path: "/", 
      label: "Головна", 
      icon: Home, 
      active: isActive("/") 
    },
    { 
      path: "/recognizer", 
      label: "Розпізнавання", 
      icon: Mic, 
      active: isActive("/recognizer") 
    },
    { 
      path: "/training", 
      label: "Тренування", 
      icon: Ear, 
      disabled: true 
    },
    { 
      path: "/testing", 
      label: "Тестування", 
      icon: BookOpenCheck, 
      disabled: true 
    }
  ];

  return (
    <header className="header">
      <div className="header__container">
        <Link to="/" className="header__logo">
          <Music className="header__logo-icon" />
          <span className="header__logo-text">HPT</span>
        </Link>
        
        <nav className="header__nav">
          {navItems.map((item) => (
            item.disabled ? (
              <button 
                key={item.path}
                className="header__nav-item header__nav-item--disabled"
                disabled
              >
                <item.icon className="header__nav-icon" />
                <span>{item.label}</span>
              </button>
            ) : (
              <Link
                key={item.path}
                to={item.path}
                className={`header__nav-item ${item.active ? 'header__nav-item--active' : ''}`}
              >
                <item.icon className="header__nav-icon" />
                <span>{item.label}</span>
              </Link>
            )
          ))}
        </nav>

        <div className="header__actions">
          <Link to="/logout" className="header__logout-btn">
            <LogOut className="header__logout-icon" />
            <span className="header__logout-text">Вийти</span>
          </Link>
        </div>
      </div>
    </header>
  );
}

export default Header;