import { useState } from "react";
import api from "../api";
import { useNavigate, Link } from "react-router-dom";
import { User, Lock, LogIn, UserPlus, AlertCircle } from "lucide-react";
import { ACCESS_TOKEN, REFRESH_TOKEN } from "../constants";
import LoadingIndicator from "./LoadingIndicator";

function Form({ route, method }) {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const navigate = useNavigate();

    const isLogin = method === "login";
    const title = isLogin ? "Вітаємо знову" : "Створити акаунт";
    const subtitle = isLogin 
        ? "Увійдіть у свій акаунт для продовження" 
        : "Створіть новий акаунт для початку роботи";
    const buttonText = isLogin ? "Увійти" : "Зареєструватися";
    const linkText = isLogin 
        ? "Ще немає акаунта?" 
        : "Вже маєте акаунт?";
    const linkAction = isLogin ? "Зареєструватися" : "Увійти";
    const linkTo = isLogin ? "/register" : "/login";

    const validateForm = () => {
        if (!username.trim()) {
            setError("Будь ласка, введіть логін");
            return false;
        }
        if (!password.trim()) {
            setError("Будь ласка, введіть пароль");
            return false;
        }
        if (password.length < 5) {
            setError("Пароль повинен містити мінімум 5 символів");
            return false;
        }
        return true;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");

        if (!validateForm()) {
            return;
        }

        setLoading(true);

        try {
            const res = await api.post(route, { username, password });
            
            if (isLogin) {
                localStorage.setItem(ACCESS_TOKEN, res.data.access);
                localStorage.setItem(REFRESH_TOKEN, res.data.refresh);
                navigate("/");
            } else {
                navigate("/login");
            }
        } catch (error) {
            console.error("Form submission error:", error);
            
            if (error.response?.data) {
                const errorData = error.response.data;
                if (errorData.username) {
                    setError("Цей логін вже використовується");
                } else if (errorData.password) {
                    setError("Невірний формат паролю");
                } else if (errorData.detail || errorData.non_field_errors) {
                    setError(isLogin ? "Невірний логін або пароль" : "Помилка реєстрації");
                } else {
                    setError("Виникла помилка. Спробуйте пізніше");
                }
            } else {
                setError("Помилка з'єднання. Перевірте інтернет");
            }
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="form-container">
            <div className="form-card">
                <div className="form-header">
                    <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem'}}>
                        {isLogin ? (
                            <LogIn style={{width: '2rem', height: '2rem'}} />
                        ) : (
                            <UserPlus style={{width: '2rem', height: '2rem'}} />
                        )}
                    </div>
                    <h1>{title}</h1>
                    <p>{subtitle}</p>
                </div>

                <form onSubmit={handleSubmit} className="form-body">
                    {error && (
                        <div className="alert alert--error">
                            <AlertCircle style={{width: '1rem', height: '1rem', flexShrink: 0, marginTop: '0.125rem'}} />
                            <div>
                                <div className="alert__title">Помилка</div>
                                <div className="alert__description">{error}</div>
                            </div>
                        </div>
                    )}

                    <div className="form-group">
                        <label htmlFor="username" className="form-label">
                            Логін
                        </label>
                        <div className="input-group">
                            <div className="input-group__addon">
                                <User style={{width: '1.125rem', height: '1.125rem'}} />
                            </div>
                            <input
                                id="username"
                                className={`form-input ${error && !username ? 'error' : ''}`}
                                type="text"
                                value={username}
                                onChange={(e) => {
                                    setUsername(e.target.value);
                                    if (error) setError("");
                                }}
                                placeholder="Введіть ваш логін"
                                disabled={loading}
                                autoComplete="username"
                                required
                            />
                        </div>
                    </div>

                    <div className="form-group">
                        <label htmlFor="password" className="form-label">
                            Пароль
                        </label>
                        <div className="input-group">
                            <div className="input-group__addon">
                                <Lock style={{width: '1.125rem', height: '1.125rem'}} />
                            </div>
                            <input
                                id="password"
                                className={`form-input ${error && !password ? 'error' : ''}`}
                                type="password"
                                value={password}
                                onChange={(e) => {
                                    setPassword(e.target.value);
                                    if (error) setError("");
                                }}
                                placeholder="Введіть ваш пароль"
                                disabled={loading}
                                autoComplete={isLogin ? "current-password" : "new-password"}
                                required
                            />
                        </div>
                        {!isLogin && (
                            <div className="form-help">
                                Пароль повинен містити мінімум 5 символів
                            </div>
                        )}
                    </div>

                    <div className="form-group">
                        {loading ? (
                            <button 
                                className="form-button btn-loading"
                                type="submit"
                                disabled
                            >
                                <LoadingIndicator size="small" />
                            </button>
                        ) : (
                            <button 
                                className="form-button"
                                type="submit"
                            >
                                {isLogin ? <LogIn /> : <UserPlus />}
                                <span>{buttonText}</span>
                            </button>
                        )}
                    </div>
                </form>

                <div className="form-footer">
                    <p>{linkText}</p>
                    <Link to={linkTo} className="font-medium">
                        {linkAction}
                    </Link>
                </div>
            </div>
        </div>
    );
}

export default Form;