import { Navigate, useLocation } from "react-router-dom";
import { useUserProfileContext } from "../contexts/UserProfileContext";

const EXPERIMENTER_ROLE = "experimenter";

/**
 * Renders children only if user is authenticated and has one of the allowed roles.
 * Must be used inside ProtectedRoute (or after auth) and UserProfileProvider.
 */
function RoleProtectedRoute({ children, allowedRoles = [EXPERIMENTER_ROLE] }) {
  const { profile, isLoading } = useUserProfileContext();
  const location = useLocation();

  if (isLoading) {
    return <div>Loading...</div>;
  }

  const role = profile?.role;
  const hasRole = role && allowedRoles.includes(role);

  if (!hasRole) {
    return <Navigate to="/" state={{ from: location }} replace />;
  }

  return children;
}

export default RoleProtectedRoute;
export { EXPERIMENTER_ROLE };
