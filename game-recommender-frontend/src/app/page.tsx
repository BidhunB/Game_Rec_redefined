"use client";
import { useState } from "react";
import ColdStart from "@/components/ColdStart";
import TFIDFRec from "@/components/TFIDFRec";
import BERTRec from "@/components/BERTRec";
import HybridBERTRec from "@/components/HybridBERTRec";
import CollaborativeRec from "@/components/CollaborativeRec";
import HybridTFIDFRec from "@/components/HybridTFIDFRec";
import UserProfile from "@/components/UserProfile";
import SignInButton from "@/components/SignInButton";
import UserStats from "@/components/UserStats";
import UserDebug from "@/components/UserDebug";
import ApiTest from "@/components/ApiTest";
import AdminSetup from "@/components/AdminSetup";
import { useAuth } from "@/contexts/AuthContext";

export default function Home() {
  const [activeTab, setActiveTab] = useState("cold-start");
  const { isAuthenticated, loading } = useAuth();

  const tabs = [
    {
      id: "cold-start",
      name: "Popular Games",
      description: "Trending and highly-rated games",
      component: <ColdStart />,
      icon: "🔥"
    },
    {
      id: "tfidf",
      name: "TF-IDF AI",
      description: "Content-based recommendations",
      component: <TFIDFRec />,
      icon: "🧠"
    },
    {
      id: "bert",
      name: "BERT AI",
      description: "Deep learning semantic analysis",
      component: <BERTRec />,
      icon: "🤖"
    },
    {
      id: "hybrid-bert",
      name: "Hybrid BERT",
      description: "Collaborative + semantic AI",
      component: <HybridBERTRec />,
      icon: "🧬"
    },
    {
      id: "collaborative",
      name: "Collaborative",
      description: "User-based filtering",
      component: <CollaborativeRec />,
      icon: "🤝"
    },
    {
      id: "hybrid-tfidf",
      name: "Hybrid TFIDF",
      description: "Collaborative + content-based AI",
      component: <HybridTFIDFRec />,
      icon: "🧬"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header with Auth */}
      <div className="relative z-20 px-6 py-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <div className="inline-flex items-center justify-center w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full shadow-lg">
              <span className="text-xl">🎮</span>
            </div>
            <h1 className="text-2xl font-bold text-white">Game Recommender</h1>
          </div>
          <div className="flex items-center space-x-4">
            {loading ? (
              <div className="text-white">Loading...</div>
            ) : (
              <>
                <UserProfile />
                <SignInButton />
              </>
            )}
          </div>
        </div>
      </div>

      {/* Hero Section */}
      <div className="relative overflow-hidden pt-20">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-pink-600/20"></div>
        
        
        {/* Animated background elements */}
        <div className="absolute top-20 left-10 w-72 h-72 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
        <div className="absolute top-40 right-10 w-72 h-72 bg-yellow-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
        <div className="absolute -bottom-8 left-20 w-72 h-72 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
      </div>

      {/* Navigation Tabs */}
      <div className="relative z-20 px-6 -mt-8">
        <div className="max-w-7xl mx-auto">
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-2 shadow-2xl border border-white/20">
            <div className="flex flex-wrap justify-center gap-2">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-3 px-6 py-4 rounded-xl font-semibold transition-all duration-300 ${
                    activeTab === tab.id
                      ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg transform scale-105"
                      : "text-gray-300 hover:text-white hover:bg-white/10"
                  }`}
                >
                  <span className="text-xl">{tab.icon}</span>
                  <div className="text-left">
                    <div className="font-bold">{tab.name}</div>
                    <div className="text-xs opacity-80">{tab.description}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Content Area */}
      <div className="relative z-10 px-6 py-12">
        <div className="max-w-7xl mx-auto">
          
          {/* Debug Info (admin only) */}
          <div className="mb-8 space-y-4">
            <UserDebug />
            <ApiTest />
          </div>
          
          {/* User Stats */}
          {isAuthenticated && (
            <div className="mb-8">
              <UserStats />
            </div>
          )}
          
          {/* Active Tab Content */}
          {tabs.find(tab => tab.id === activeTab)?.component}
        </div>
      </div>

      {/* Footer */}
      <div className="relative z-10 px-6 py-8 mt-16">
        <div className="max-w-7xl mx-auto text-center">
          <div className="border-t border-white/10 pt-8">
            <p className="text-gray-400 text-sm">
              Powered by advanced AI algorithms • Built with Next.js & Tailwind CSS
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}