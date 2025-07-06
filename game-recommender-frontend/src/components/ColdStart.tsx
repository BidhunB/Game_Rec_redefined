"use client";
import { useEffect, useState } from "react";
import LoadingSpinner from "./LoadingSpinner";
import LikeButton from "./LikeButton";

type Game = {
  id: number;
  name: string;
  genre_text: string;
  rating?: number;
  ratings_count?: number;
  background_image?: string;
};

export default function ColdStart() {
  const [games, setGames] = useState<Game[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://localhost:8000/cold-start") // your FastAPI route
      .then((res) => res.json())
      .then((data) => {
        setGames(data);
        console.log(data);
        setLoading(false);
      });
  }, []);

  if (loading) return <LoadingSpinner color="blue" text="Loading popular games..." />;

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-white mb-2">
          🔥 Popular Games
        </h2>
        <p className="text-gray-300">Discover the most trending and highly-rated games</p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {games.map((game) => (
          <div
            key={game.id}
            className="group bg-white/10 backdrop-blur-sm rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-500 border border-white/20 overflow-hidden hover:-translate-y-2"
          >
            {game.background_image && (
              <div className="relative h-48 overflow-hidden">
                <img
                  src={game.background_image}
                  alt={game.name}
                  className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent"></div>
                <div className="absolute top-3 right-3">
                  <div className="bg-black/60 backdrop-blur-sm text-white px-3 py-1 rounded-full text-sm font-medium">
                    ⭐ {game.rating?.toFixed(1) || 'N/A'}
                  </div>
                </div>
              </div>
            )}
            
            <div className="p-5">
              <h3 className="font-bold text-lg mb-3 text-white group-hover:text-blue-300 transition-colors duration-300 line-clamp-2">
                {game.name}
              </h3>
              
              <div className="mb-4">
                <div className="flex flex-wrap gap-2">
                  {game.genre_text.split(',').map((genre, index) => (
                    <span
                      key={index}
                      className="bg-blue-500/20 text-blue-200 px-3 py-1 rounded-full text-xs font-medium border border-blue-400/30"
                    >
                      {genre.trim()}
                    </span>
                  ))}
                </div>
              </div>
              
              <div className="flex items-center justify-between pt-3 border-t border-white/10">
                <div className="flex items-center space-x-2">
                  <div className="flex items-center space-x-1">
                    <span className="text-yellow-400 text-lg">⭐</span>
                    <span className="text-sm font-semibold text-gray-200">
                      {game.rating?.toFixed(1) || 'N/A'}
                    </span>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="flex items-center space-x-1">
                    <span className="text-blue-400 text-lg">👍</span>
                    <span className="text-sm text-gray-300 font-medium">
                      {game.ratings_count ? (game.ratings_count > 1000 ? `${(game.ratings_count / 1000).toFixed(1)}K` : game.ratings_count.toLocaleString()) : 'N/A'}
                    </span>
                  </div>
                </div>
              </div>
              
              {/* Like Button */}
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
