"use client";
import { useAuth } from "@/contexts/AuthContext";

export default function SignInButton() {
  const { signIn, isAuthenticated } = useAuth();

  if (isAuthenticated) {
    return null;
  }

  return (
    <button
      onClick={signIn}
      className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-semibold px-6 py-3 rounded-xl transition-all duration-300 hover:scale-105 shadow-lg"
    >
      Sign In
    </button>
  );
} 