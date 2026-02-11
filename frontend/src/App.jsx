import react from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Home from "./pages/Home";
import NotFound from "./pages/NotFound";
import ProtectedRoute from "./components/ProtectedRoute";
import AudioRecognizer from "./pages/AudioRecognizer";
import Training from "./pages/Training";
import Testing from "./pages/Testing";
import Experiments from "./pages/Experiments";
import ExperimentOvertones from "./pages/ExperimentOvertones";
import { UserProfileProvider } from "./contexts/UserProfileContext";
import RoleProtectedRoute from "./components/RoleProtectedRoute";
import './styles/main.scss';

function Logout() {
  localStorage.clear();
  return <Navigate to="/login" />;
}

function RegisterAndLogout() {
  localStorage.clear();
  return <Register />;
}

function App() {
  return (
    <UserProfileProvider>
      <BrowserRouter>
        <Routes>
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Home />
              </ProtectedRoute>
            }
          />
          <Route
            path="/recognizer"
            element={
              <ProtectedRoute>
                <AudioRecognizer />
              </ProtectedRoute>
            }
          />
          <Route
            path="/training"
            element={
              <ProtectedRoute>
                <Training />
              </ProtectedRoute>
            }
          />
          <Route
            path="/testing"
            element={
              <ProtectedRoute>
                <Testing />
              </ProtectedRoute>
            }
          />
          <Route
            path="/experiments"
            element={
              <ProtectedRoute>
                <RoleProtectedRoute>
                  <Experiments />
                </RoleProtectedRoute>
              </ProtectedRoute>
            }
          />
          <Route
            path="/experiments/overtones"
            element={
              <ProtectedRoute>
                <RoleProtectedRoute>
                  <ExperimentOvertones />
                </RoleProtectedRoute>
              </ProtectedRoute>
            }
          />
          <Route path="/login" element={<Login />} />
          <Route path="/logout" element={<Logout />} />
          <Route path="/register" element={<RegisterAndLogout />} />
          <Route path="*" element={<NotFound />}></Route>
        </Routes>
      </BrowserRouter>
    </UserProfileProvider>
  )
}

export default App
