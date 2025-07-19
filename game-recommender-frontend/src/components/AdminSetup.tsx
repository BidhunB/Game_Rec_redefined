"use client";
import { useAuth } from "@/contexts/AuthContext";
import { getAdminInfo } from "@/utils/userUtils";

export default function AdminSetup() {
  const { user, isAuthenticated, loading } = useAuth();
  
  if (loading) {
    return null;
  }

  if (!isAuthenticated || !user) {
    return null;
  }

  const adminInfo = getAdminInfo(user);

  return (
    <div className="bg-purple-500/20 border border-purple-400/30 rounded-lg p-4 mb-4">
      <h3 className="text-purple-300 font-semibold mb-2">🔧 Admin Access</h3>
      <div className="text-purple-200 text-sm space-y-2">
        <p><strong>Current Email:</strong> {user.email}</p>
        <p><strong>Admin Status:</strong> {adminInfo.isAdmin ? "✅ Admin" : "❌ Not Admin"}</p>
        <p><strong>Reason:</strong> {adminInfo.reason}</p>
        
        {!adminInfo.isAdmin && (
          <div className="mt-3 p-3 bg-yellow-500/20 border border-yellow-400/30 rounded">
            <p className="text-yellow-200 text-xs">
              <strong>To enable admin access:</strong>
            </p>
            <ol className="text-yellow-200 text-xs mt-2 space-y-1 list-decimal list-inside">
              <li>Open <code className="bg-black/30 px-1 rounded">src/utils/userUtils.ts</code></li>
              <li>Find the <code className="bg-black/30 px-1 rounded">adminEmails</code> array</li>
              <li>Add your email: <code className="bg-black/30 px-1 rounded">"{user.email}"</code></li>
              <li>Save the file and restart the development server</li>
            </ol>
          </div>
        )}
      </div>
    </div>
  );
} 