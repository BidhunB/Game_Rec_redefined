"use client";
import { useState } from "react";
import { useToast } from "@/contexts/ToastContext";

interface LikeButtonProps {
  gameId: number;
  gameName: string;
  initialLiked?: boolean;
}

export default function LikeButton({ gameId, gameName, initialLiked = false }: LikeButtonProps) {
  const [isLiked, setIsLiked] = useState(initialLiked);
  const [isLoading, setIsLoading] = useState(false);
  const { showToast } = useToast();

  const handleLike = async () => {
    setIsLoading(true);
    try {
      const response = await fetch("http://localhost:8000/like", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          game_id: gameId,
          game_name: gameName,
          action: isLiked ? "unlike" : "like",
          timestamp: new Date().toISOString(),
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setIsLiked(!isLiked);
        showToast(data.message || `Successfully ${isLiked ? 'unliked' : 'liked'} ${gameName}`, "success");
      } else {
        showToast("Failed to update like status", "error");
      }
    } catch (error) {
      console.error("Error updating like status:", error);
      showToast("Network error. Please try again.", "error");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <button
      onClick={handleLike}
      disabled={isLoading}
      className={`flex items-center space-x-2 px-4 py-2 rounded-full transition-all duration-300 ${
        isLiked
          ? "bg-red-500/20 text-red-400 border border-red-400/30 hover:bg-red-500/30"
          : "bg-white/10 text-gray-300 border border-white/20 hover:bg-white/20 hover:text-white"
      } ${isLoading ? "opacity-50 cursor-not-allowed" : "hover:scale-105"}`}
    >
      {isLoading ? (
        <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
      ) : (
        <span className="text-lg">{isLiked ? "❤️" : "🤍"}</span>
      )}
      <span className="text-sm font-medium">
        {isLoading ? "..." : isLiked ? "Liked" : "Like"}
      </span>
    </button>
  );
} 