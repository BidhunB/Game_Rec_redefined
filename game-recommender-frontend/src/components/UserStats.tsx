"use client";
import { useEffect, useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { getUserId, sanitizeUserId } from "@/utils/userUtils";

interface UserStats {
  total_interactions: number;
  liked_games: number;
  disliked_games: number;
  avg_rating: number;
  max_rating: number;
  min_rating: number;
}

export default function UserStats() {
  const [stats, setStats] = useState<UserStats | null>(null);
  const [loading, setLoading] = useState(true);
  const { user, isAuthenticated } = useAuth();

  useEffect(() => {
    if (!isAuthenticated || !user) {
      setLoading(false);
      return;
    }

    const userId = sanitizeUserId(getUserId(user, "anonymous"));
    console.log("[UserStats] Fetching stats for user:", userId);
    
    fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/user/stats/${encodeURIComponent(userId)}`)
      .then((res) => res.json())
      .then((data) => {
        if (data.success) {
          setStats(data.stats);
        }
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching user stats:", error);
        setLoading(false);
      });
  }, [user, isAuthenticated]);

  if (!isAuthenticated) {
    return null;
  }

  if (loading) {
    return (
      <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
        <div className="animate-pulse">
          <div className="h-4 bg-white/20 rounded mb-4"></div>
          <div className="grid grid-cols-2 gap-4">
            <div className="h-8 bg-white/20 rounded"></div>
            <div className="h-8 bg-white/20 rounded"></div>
            <div className="h-8 bg-white/20 rounded"></div>
            <div className="h-8 bg-white/20 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
        <h3 className="text-lg font-semibold text-white mb-4">Your Stats</h3>
        <p className="text-gray-300">No interactions yet. Start rating games to see your stats!</p>
      </div>
    );
  }

  return (
    <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
      <h3 className="text-lg font-semibold text-white mb-4">Your Stats</h3>
      <div className="grid grid-cols-2 gap-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-400">{stats.total_interactions}</div>
          <div className="text-sm text-gray-300">Total Ratings</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-green-400">{stats.liked_games}</div>
          <div className="text-sm text-gray-300">Liked Games</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-red-400">{stats.disliked_games}</div>
          <div className="text-sm text-gray-300">Disliked Games</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-yellow-400">{stats.avg_rating.toFixed(1)}</div>
          <div className="text-sm text-gray-300">Avg Rating</div>
        </div>
      </div>
      <div className="mt-4 pt-4 border-t border-white/10">
        <div className="flex justify-between text-sm text-gray-300">
          <span>Rating Range: {stats.min_rating} - {stats.max_rating}</span>
        </div>
      </div>
    </div>
  );
} 