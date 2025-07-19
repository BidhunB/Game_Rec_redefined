"use client";
import { useAuth } from "@/contexts/AuthContext";
import { isAdmin } from "@/utils/userUtils";
import Image from "next/image";

export default function UserProfile() {
  const { user, signOut, isAuthenticated } = useAuth();

  if (!isAuthenticated || !user) {
    return null;
  }

  const adminStatus = isAdmin(user);

  return (
    <div className="flex items-center space-x-4">
      <div className="flex items-center space-x-3 bg-white/10 backdrop-blur-sm rounded-xl px-4 py-2 border border-white/20">
        {user.image && (
          <div className="relative w-8 h-8 rounded-full overflow-hidden">
            <Image
              src={user.image}
              alt={user.name}
              fill
              className="object-cover"
            />
          </div>
        )}
        <div className="text-white">
          <div className="text-sm font-medium flex items-center gap-2">
            {user.name}
            {adminStatus && (
              <span className="bg-purple-500/80 text-white px-2 py-0.5 rounded-full text-xs font-bold">
                ADMIN
              </span>
            )}
          </div>
          <div className="text-xs text-gray-300">{user.email}</div>
        </div>
      </div>
      <button
        onClick={signOut}
        className="bg-red-500/20 hover:bg-red-500/30 text-red-300 hover:text-red-200 px-4 py-2 rounded-xl border border-red-400/30 transition-all duration-300 hover:scale-105"
      >
        Sign Out
      </button>
    </div>
  );
} 