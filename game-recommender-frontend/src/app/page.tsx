"use client";
import { useState } from "react";
import ColdStart from "@/components/ColdStart";
import TFIDFRec from "@/components/TFIDFRec";
import BERTRec from "@/components/BERTRec";
import HybridBERTRec from "@/components/HybridBERTRec";
import CollaborativeRec from "@/components/CollaborativeRec";
import HybridTFIDFRec from "@/components/HybridTFIDFRec";

export default function Home() {
  const [activeTab, setActiveTab] = useState("cold-start");

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
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-pink-600/20"></div>
        <div className="relative z-10 px-6 py-16">
          <div className="max-w-7xl mx-auto text-center">
            <div className="mb-8">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full mb-6 shadow-2xl">
                <span className="text-3xl">🎮</span>
              </div>
              <h1 className="text-5xl md:text-7xl font-bold bg-gradient-to-r from-white via-blue-100 to-purple-100 bg-clip-text text-transparent mb-4">
                Game Recommender
              </h1>
              <p className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
                Discover your next favorite game with AI-powered recommendations
              </p>
            </div>
          </div>
        </div>
        
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