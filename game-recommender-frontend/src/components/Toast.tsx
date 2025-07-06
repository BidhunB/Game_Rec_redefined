"use client";
import { useEffect, useState } from "react";

interface ToastProps {
  message: string;
  type: "success" | "error" | "info";
  isVisible: boolean;
  onClose: () => void;
}

export default function Toast({ message, type, isVisible, onClose }: ToastProps) {
  useEffect(() => {
    if (isVisible) {
      const timer = setTimeout(() => {
        onClose();
      }, 3000);

      return () => clearTimeout(timer);
    }
  }, [isVisible, onClose]);

  if (!isVisible) return null;

  const typeClasses = {
    success: "bg-green-500/20 border-green-400/30 text-green-300",
    error: "bg-red-500/20 border-red-400/30 text-red-300",
    info: "bg-blue-500/20 border-blue-400/30 text-blue-300"
  };

  const icons = {
    success: "✅",
    error: "❌",
    info: "ℹ️"
  };

  return (
    <div className="fixed bottom-8 left-1/2 transform -translate-x-1/2 z-50 animate-slide-up">
      <div className={`${typeClasses[type]} backdrop-blur-sm border rounded-lg px-6 py-4 shadow-2xl flex items-center space-x-3 min-w-80 max-w-md`}>
        <span className="text-xl">{icons[type]}</span>
        <span className="text-sm font-medium flex-1 text-center">{message}</span>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-white transition-colors ml-2"
        >
          ✕
        </button>
      </div>
    </div>
  );
} 