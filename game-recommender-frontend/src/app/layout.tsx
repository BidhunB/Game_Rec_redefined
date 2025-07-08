import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ToastProvider } from "@/contexts/ToastContext";
import type { Viewport } from 'next'

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Game Recommender - AI-Powered Game Discovery",
  description: "Discover your next favorite game with AI-powered recommendations using TF-IDF and BERT algorithms",
  keywords: "game recommender, AI, machine learning, TF-IDF, BERT, game discovery",
  authors: [{ name: "Game Recommender Team" }],
};
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body
        className={`${inter.variable} font-sans antialiased`}
      >
        <ToastProvider>
          {children}
        </ToastProvider>
      </body>
    </html>
  );
}
