"use client";
import { useAuth } from "@/contexts/AuthContext";
import { getUserId, isValidUserId } from "@/utils/userUtils";

interface EmptyStateProps {
  algorithm: string;
  color: string;
  onRetry?: () => void;
}

export default function EmptyState({ algorithm, color, onRetry }: EmptyStateProps) {
  const { user, isAuthenticated } = useAuth();
  const userId = getUserId(user, "user1");
  const isValidUser = isValidUserId(userId);

  const getEmptyStateMessage = () => {
    if (!isAuthenticated) {
      return {
        title: "Sign in to get personalized recommendations",
        description: "Create an account to receive AI-powered game suggestions based on your preferences",
        action: "Sign in to continue"
      };
    }

    if (!isValidUser) {
      return {
        title: "No user data available",
        description: "We need more information about your preferences to provide recommendations",
        action: "Try interacting with some games first"
      };
    }

    return {
      title: `No ${algorithm} recommendations available`,
      description: "We couldn't find any games matching your preferences with this algorithm. Try liking or disliking some games to improve recommendations.",
      action: "Explore other recommendation types"
    };
  };

  const message = getEmptyStateMessage();

  return (
    <div className="text-center py-12">
      <div className="max-w-md mx-auto">
        {/* Icon */}
        <div className={`inline-flex items-center justify-center w-20 h-20 bg-${color}-500/20 rounded-full mb-6 border border-${color}-400/30`}>
          <span className="text-3xl">
            {!isAuthenticated ? "🔐" : !isValidUser ? "📊" : "🎯"}
          </span>
        </div>
        
        {/* Title */}
        <h3 className={`text-xl font-semibold text-${color}-300 mb-3`}>
          {message.title}
        </h3>
        
        {/* Description */}
        <p className="text-gray-300 mb-6 leading-relaxed">
          {message.description}
        </p>
        
        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          {onRetry && (
            <button
              onClick={onRetry}
              className={`bg-${color}-500/80 hover:bg-${color}-600/90 text-white px-6 py-3 rounded-lg font-medium transition-colors duration-300`}
            >
              🔄 Try Again
            </button>
          )}
          
          {!isAuthenticated && (
            <a
              href="/auth/signin"
              className={`bg-${color}-500/80 hover:bg-${color}-600/90 text-white px-6 py-3 rounded-lg font-medium transition-colors duration-300 inline-block`}
            >
              🔐 Sign In
            </a>
          )}
        </div>
        
        {/* Additional Info */}
        {isAuthenticated && isValidUser && (
          <div className="mt-6 p-4 bg-gray-800/50 rounded-lg border border-gray-700/50">
            <p className="text-sm text-gray-400 mb-2">
              <strong>Tips to get better recommendations:</strong>
            </p>
            <ul className="text-sm text-gray-400 space-y-1 text-left">
              <li>• Like or dislike games you've played</li>
              <li>• Try different recommendation algorithms</li>
              <li>• Explore games from various genres</li>
              <li>• Check back later as we learn your preferences</li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
} 