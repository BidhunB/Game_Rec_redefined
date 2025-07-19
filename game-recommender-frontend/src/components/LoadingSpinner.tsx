"use client";
import React from 'react';

interface LoadingSpinnerProps {
  color?: string;
  text?: string;
}

export default function LoadingSpinner({ color = "blue", text = "Loading..." }: LoadingSpinnerProps) {
  const colorClasses = {
    blue: "text-blue-500",
    green: "text-green-500",
    red: "text-red-500",
    yellow: "text-yellow-500",
    purple: "text-purple-500",
    white: "text-white"
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-[400px]">
      <div className={`animate-spin rounded-full h-12 w-12 border-b-2 border-t-2 ${colorClasses[color as keyof typeof colorClasses] || colorClasses.blue}`}></div>
      <p className="mt-4 text-gray-300 text-lg">{text}</p>
    </div>
  );
} 