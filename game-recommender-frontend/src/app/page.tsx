import ColdStart from "@/components/ColdStart";
import TFIDFRec from "@/components/TFIDFRec";
import BERTRec from "@/components/BERTRec";

export default function Home() {
  return (
    <main className="min-h-screen p-6">
      <h1 className="text-3xl font-bold mb-6">🎮 Game Recommender</h1>
        <ColdStart />
        <TFIDFRec />
        <BERTRec />
    </main>

  );
}