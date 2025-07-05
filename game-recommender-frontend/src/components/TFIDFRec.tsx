"use client";
import { useEffect, useState } from "react";

type Game = {
  id: number;
  name: string;
  genre_text: string;
  similarity: number;
};

export default function TFIDFRec() {
  const [games, setGames] = useState<Game[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/recommend/tfidf?user_id=user1")
      .then((res) => res.json())
      .then((data) => {
        setGames(data);
        setLoading(false);
      });
  }, []);

  if (loading) return <p>Loading TF-IDF recommendations...</p>;

  return (
    <div className="p-4 mt-8">
      <h2 className="text-2xl font-bold mb-4">🧠 TF-IDF Recommendations</h2>
      <ul className="space-y-3">
        {games.map((game) => (
          <li key={game.id} className="p-3 border rounded shadow hover:bg-gray-50 transition">
            <p className="font-semibold">{game.name}</p>
            <p className="text-sm text-gray-600">Genres: {game.genre_text}</p>
            <p className="text-sm">📈 Similarity: {game.similarity.toFixed(3)}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
