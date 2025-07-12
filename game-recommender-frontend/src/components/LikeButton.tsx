"use client";
import { useState } from "react";
import { useToast } from "@/contexts/ToastContext";
import { createPortal } from "react-dom";

interface LikeButtonProps {
  gameId: number;
  gameName: string;
  initialLiked?: boolean;
  initialDisliked?: boolean;
}

// RatingPopup component
function RatingPopup({ onSubmit, onCancel, isLoading }: { onSubmit: (rating: number) => void; onCancel: () => void; isLoading: boolean }) {
  const [selectedRating, setSelectedRating] = useState<number | null>(null);

  const popup = (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl p-8 shadow-2xl w-80 flex flex-col items-center">
        <h3 className="text-xl font-bold mb-4 text-white">Rate this game</h3>
        <div className="flex space-x-2 mb-6">
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              type="button"
              className={`text-3xl transition-colors ${selectedRating && star <= selectedRating ? 'text-yellow-400 drop-shadow' : 'text-gray-500 hover:text-yellow-300'}`}
              onClick={() => setSelectedRating(star)}
              aria-label={`Rate ${star} star${star > 1 ? 's' : ''}`}
            >
              ★
            </button>
          ))}
        </div>
        <div className="flex space-x-4 w-full justify-center">
          <button
            className="px-4 py-2 rounded-lg bg-white/20 text-gray-200 border border-white/20 hover:bg-white/30 transition-colors"
            onClick={onCancel}
            disabled={isLoading}
          >
            Cancel
          </button>
          <button
            className={`px-4 py-2 rounded-lg bg-blue-500/80 text-white font-semibold border border-blue-400/40 hover:bg-blue-600/90 transition-colors ${(!selectedRating || isLoading) ? 'opacity-50 cursor-not-allowed' : ''}`}
            onClick={() => selectedRating && onSubmit(selectedRating)}
            disabled={!selectedRating || isLoading}
          >
            {isLoading ? 'Submitting...' : 'Submit'}
          </button>
        </div>
      </div>
    </div>
  );

  if (typeof window === 'undefined') return null;
  return createPortal(popup, document.body);
}

export default function LikeButton({ gameId, gameName, initialLiked = false, initialDisliked = false }: LikeButtonProps) {
  const [isLiked, setIsLiked] = useState(initialLiked);
  const [isDisliked, setIsDisliked] = useState(initialDisliked);
  const [isLoading, setIsLoading] = useState(false);
  const [showRating, setShowRating] = useState<null | 'like' | 'dislike'>(null);
  const { showToast } = useToast();

  const handleLike = () => {
    setShowRating('like');
  };
  const handleDislike = () => {
    setShowRating('dislike');
  };

  const handleRatingSubmit = async (rating: number) => {
    setIsLoading(true);
    const payload = {
      game_id: gameId,
      game_name: gameName,
      action: showRating === 'like' ? (isLiked ? "unlike" : "like") : (isDisliked ? "un-dislike" : "dislike"),
      rating,
      timestamp: new Date().toISOString(),
    };
    try {
      // Also submit to /newInteraction
      const interactionPayload = {
        user_id: "user1",
        game_id: gameId,
        liked: showRating === 'like',
        rating,
        timestamp: new Date().toISOString(),
      };
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/newInteraction`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(interactionPayload),
      });

      if (response.ok) {
        const data = await response.json();
        if (showRating === 'like') {
          setIsLiked(!isLiked);
          if (!isLiked && isDisliked) setIsDisliked(false); // mutually exclusive
        } else {
          setIsDisliked(!isDisliked);
          if (!isDisliked && isLiked) setIsLiked(false); // mutually exclusive
        }
        showToast(data.message || `Successfully ${showRating === 'like' ? (isLiked ? 'unliked' : 'liked') : (isDisliked ? 'un-disliked' : 'disliked')} ${gameName} with rating ${rating}`, "success");
      } else {
        showToast("Failed to update status", "error");
      }
    } catch (error) {
      console.error("Error updating status:", error);
      showToast("Network error. Please try again.", "error");
    } finally {
      setIsLoading(false);
      setShowRating(null);
    }
  };

  return (
    <>
      <div className="flex items-center space-x-3">
        <button
          onClick={handleLike}
          disabled={isLoading}
          className={`flex items-center space-x-2 px-4 py-2 rounded-full transition-all duration-300 border ${
            isLiked
              ? "bg-red-500/20 text-red-400 border-red-400/30 hover:bg-red-500/30"
              : "bg-white/10 text-gray-300 border-white/20 hover:bg-white/20 hover:text-white"
          } ${isLoading ? "opacity-50 cursor-not-allowed" : "hover:scale-105"}`}
        >
          {isLoading && showRating === 'like' ? (
            <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
          ) : (
            <span className="text-lg">{isLiked ? "❤️" : "🤍"}</span>
          )}
          <span className="text-sm font-medium">
            {isLoading && showRating === 'like' ? "..." : isLiked ? "Liked" : "Like"}
          </span>
        </button>
        <button
          onClick={handleDislike}
          disabled={isLoading}
          className={`flex items-center space-x-2 px-4 py-2 rounded-full transition-all duration-300 border ${
            isDisliked
              ? "bg-blue-500/20 text-blue-400 border-blue-400/30 hover:bg-blue-500/30"
              : "bg-white/10 text-gray-300 border-white/20 hover:bg-white/20 hover:text-white"
          } ${isLoading ? "opacity-50 cursor-not-allowed" : "hover:scale-105"}`}
        >
          {isLoading && showRating === 'dislike' ? (
            <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
          ) : (
            <span className="text-lg">{isDisliked ? "👎" : "🙁"}</span>
          )}
          <span className="text-sm font-medium">
            {isLoading && showRating === 'dislike' ? "..." : isDisliked ? "Disliked" : "Dislike"}
          </span>
        </button>
      </div>
      {showRating && (
        <RatingPopup
          onSubmit={handleRatingSubmit}
          onCancel={() => setShowRating(null)}
          isLoading={isLoading}
        />
      )}
    </>
  );
} 