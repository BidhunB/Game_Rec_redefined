"use client";
import { useEffect, useState } from "react";

type Game = {
  id: number;
  name: string;
  genre_text: string;
  rating?: number;
  ratings_count?: number;
};

export default function ColdStart() {
  const [games, setGames] = useState<Game[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://localhost:8000/cold-start") // your FastAPI route
      .then((res) => res.json())
      .then((data) => {
        setGames(data);
        setLoading(false);
      });
  }, []);

  if (loading) return <p>Loading top-rated games...</p>;

  return (
    <div className="p-4">
      <h2 className="text-2xl font-bold mb-4">🔥 Cold Start Recommendations</h2>
      <ul className="space-y-3">
        {games.map((game) => (
          <li
            key={game.id}
            className="p-3 border rounded shadow hover:bg-gray-50 transition"
          >
            <p className="font-semibold">{game.name}</p>
            <p className="text-sm text-gray-600">Genres: {game.genre_text}</p>
            <p className="text-sm">⭐ {game.rating} | 👍 {game.ratings_count}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
