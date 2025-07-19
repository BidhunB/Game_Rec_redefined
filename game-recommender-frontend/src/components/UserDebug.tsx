"use client";
import { useAuth } from "@/contexts/AuthContext";
import { getUserId, isValidUserId, sanitizeUserId, isAdmin } from "@/utils/userUtils";

export default function UserDebug() {
  const { user, isAuthenticated, loading } = useAuth();
  
  if (loading) {
    return null;
  }

  // Only show for admin users
  if (!isAuthenticated || !user || !isAdmin(user)) {
    return null;
  }

  const userId = getUserId(user, "anonymous");
  const sanitizedUserId = sanitizeUserId(userId);
  const isValid = isValidUserId(userId);

  return (
    <div className="bg-green-500/20 border border-green-400/30 rounded-lg p-4 mb-4">
      <h3 className="text-green-300 font-semibold mb-2">🔧 Admin Debug Info</h3>
      <div className="text-green-200 text-sm space-y-1">
        <p><strong>Authentication Status:</strong> Authenticated</p>
        <p><strong>Admin Status:</strong> ✅ Admin User</p>
        <p><strong>User ID (NextAuth):</strong> {user.id || "Not set"}</p>
        <p><strong>User Email:</strong> {user.email || "Not set"}</p>
        <p><strong>User Name:</strong> {user.name || "Not set"}</p>
        <p><strong>Processed User ID:</strong> {userId}</p>
        <p><strong>Sanitized User ID:</strong> {sanitizedUserId}</p>
        <p><strong>Valid User ID:</strong> {isValid ? "Yes" : "No"}</p>
        <p><strong>User Image:</strong> {user.image ? "Available" : "Not available"}</p>
      </div>
    </div>
  );
} 