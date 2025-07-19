"use client";
import { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { getUserId, sanitizeUserId, isAdmin } from "@/utils/userUtils";

export default function ApiTest() {
  const { user, isAuthenticated, loading } = useAuth();
  const [testResults, setTestResults] = useState<string[]>([]);
  const [isTesting, setIsTesting] = useState(false);

  // Only show for admin users
  if (loading || !isAuthenticated || !user || !isAdmin(user)) {
    return null;
  }

  const addResult = (message: string) => {
    setTestResults(prev => [...prev, `${new Date().toLocaleTimeString()}: ${message}`]);
  };

  const testApi = async () => {
    setIsTesting(true);
    setTestResults([]);
    
    if (!isAuthenticated || !user || !isAdmin(user)) {
      addResult("❌ User not authenticated or not admin");
      setIsTesting(false);
      return;
    }

    const userId = sanitizeUserId(getUserId(user, "anonymous"));
    addResult(`🔍 Testing with user ID: ${userId}`);

    try {
      // Test 1: User Stats
      addResult("📊 Testing user stats endpoint...");
      const statsResponse = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/user/stats/${encodeURIComponent(userId)}`);
      if (statsResponse.ok) {
        const statsData = await statsResponse.json();
        addResult(`✅ User stats: ${statsData.stats.total_interactions} interactions`);
      } else {
        addResult(`❌ User stats failed: ${statsResponse.status}`);
      }

      // Test 2: TF-IDF Recommendations
      addResult("🧠 Testing TF-IDF recommendations...");
      const recResponse = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/recommend/tfidf?user_id=${encodeURIComponent(userId)}`);
      if (recResponse.ok) {
        const recData = await recResponse.json();
        addResult(`✅ TF-IDF recommendations: ${recData.length} games`);
      } else {
        addResult(`❌ TF-IDF recommendations failed: ${recResponse.status}`);
      }

      // Test 3: Cold Start
      addResult("🔥 Testing cold start recommendations...");
      const coldResponse = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/cold-start`);
      if (coldResponse.ok) {
        const coldData = await coldResponse.json();
        addResult(`✅ Cold start: ${coldData.length} games`);
      } else {
        addResult(`❌ Cold start failed: ${coldResponse.status}`);
      }

    } catch (error) {
      addResult(`❌ Network error: ${error}`);
    }

    setIsTesting(false);
  };

  return (
    <div className="bg-blue-500/20 border border-blue-400/30 rounded-lg p-4 mb-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-blue-300 font-semibold">🔧 Admin API Test</h3>
        <button
          onClick={testApi}
          disabled={isTesting}
          className="bg-blue-500/80 hover:bg-blue-600/90 disabled:opacity-50 text-white px-4 py-2 rounded-lg text-sm transition-colors"
        >
          {isTesting ? "Testing..." : "Test API"}
        </button>
      </div>
      
      <div className="bg-black/20 rounded p-3 max-h-40 overflow-y-auto">
        {testResults.length === 0 ? (
          <p className="text-blue-200 text-sm">Click "Test API" to run tests...</p>
        ) : (
          <div className="space-y-1">
            {testResults.map((result, index) => (
              <p key={index} className="text-blue-200 text-xs font-mono">
                {result}
              </p>
            ))}
          </div>
        )}
      </div>
    </div>
  );
} 