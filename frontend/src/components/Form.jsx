import { useState } from "react";
import api from "../api";
import { useNavigate } from "react-router-dom";
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
            const res = await api.post(route, { username, password })
            if (method === "login") {
                localStorage.setItem(ACCESS_TOKEN, res.data.access);
                localStorage.setItem(REFRESH_TOKEN, res.data.refresh);
                navigate("/")
            } else {
                navigate("/login")
            }
        } catch (error) {
            alert(error)
        } finally {
            setLoading(false)
        }
    };

    return (
        <form onSubmit={handleSubmit} className="form-container">
            <h1>{name}</h1>
            <input
                className="form-input"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Логін"
            />
            <input
                className="form-input"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Пароль"
            />
            {loading && <LoadingIndicator />}
            <button className="form-button" type="submit">
                {name}
            </button>
        </form>
    );
}

export default Form