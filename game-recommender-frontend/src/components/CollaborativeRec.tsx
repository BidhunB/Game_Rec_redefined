"use client";
import { useEffect, useState } from "react";
import LoadingSpinner from "./LoadingSpinner";
import LikeButton from "./LikeButton";
import Image from "next/image";

type Game = {
  id: number;
  name: string;
  genre_text: string;
  score: number;
  background_image?: string;
};

export default function CollaborativeRec() {
  const [games, setGames] = useState<Game[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/recommend/collaborative?user_id=user1")
      .then((res) => res.json())
      .then((data) => {
        setGames(data);
        setLoading(false);
      });
  }, []);

  if (loading) return <LoadingSpinner color="yellow" text="Loading Collaborative recommendations..." />;

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-white mb-2">
          🤝 Collaborative Filtering
        </h2>
        <p className="text-gray-300">Recommendations based on similar users' preferences</p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {games.map((game) => (
          <div
            key={game.id}
            className="group bg-white/10 backdrop-blur-sm rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-500 border border-white/20 overflow-hidden hover:-translate-y-2"
          >
            {game.background_image && (
              <div className="relative h-48 overflow-hidden">
                <Image
                  src={game.background_image}
                  alt={game.name}
                  className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700"
                  layout="fill"
                  objectFit="cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent"></div>
                <div className="absolute top-3 right-3">
                  <div className="bg-black/60 backdrop-blur-sm text-white px-3 py-1 rounded-full text-sm font-medium">
                    🏆 {game.score.toFixed(2)}
                  </div>
                </div>
              </div>
            )}
            <div className="p-5">
              <h3 className="font-bold text-lg mb-3 text-white group-hover:text-yellow-300 transition-colors duration-300 line-clamp-2">
                {game.name}
              </h3>
              <div className="mb-4">
                <div className="flex flex-wrap gap-2">
                  {game.genre_text.split(',').map((genre, index) => (
                    <span
                      key={index}
                      className="bg-yellow-500/20 text-yellow-200 px-3 py-1 rounded-full text-xs font-medium border border-yellow-400/30"
                    >
                      {genre.trim()}
                    </span>
                  ))}
                </div>
              </div>
              {/* Like Button (now with rating popup for both like and dislike) */}
              <div className="mt-4 flex justify-center">
                <LikeButton gameId={game.id} gameName={game.name} />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
} 