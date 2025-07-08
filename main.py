from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
import threading
import csv

from recommender_test import (
    load_games_dataset,
    cold_start_recommendations,
    prepare_tfidf_matrix,
    get_sample_interactions,
    recommend_for_user,
    sentence_transformer_model,
    recommend_for_user_sentence_transformer,
    hybrid_recommendation_sentence_transformer,
    get_collaborative_scores,
    hybrid_recommendation
)

app = FastAPI()

# === CORS Setup ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Constants ===
INTERACTIONS_FILE = Path("user_interactions.csv")

# Shared variables
games_df = load_games_dataset("rawg_games.csv")
tfidf_matrix = prepare_tfidf_matrix(games_df)
embeddings = sentence_transformer_model(games_df)
lock = threading.Lock()

def update_game_data():
    global games_df, tfidf_matrix, embeddings
    try:
        print("[Update] Loading fresh data...")
        new_df = load_games_dataset("rawg_games.csv")
        new_tfidf = prepare_tfidf_matrix(new_df)
        new_embeddings = sentence_transformer_model(new_df)

        with lock:
            games_df = new_df
            tfidf_matrix = new_tfidf
            embeddings = new_embeddings
        print("[Update] Game data updated.")
    except Exception as e:
        print(f"[Update Error] {e}")

# Scheduler setup
scheduler = BackgroundScheduler()
scheduler.add_job(update_game_data, "interval", minutes=5)  # Adjust timing as needed
scheduler.start()

# === Pydantic Models ===
class InteractionData(BaseModel):
    user_id: str
    game_id: int
    liked: bool
    rating: int
    timestamp: str

# === Helper Function ===
def log_interaction(data: InteractionData):
    is_new = not INTERACTIONS_FILE.exists()
    with INTERACTIONS_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["user_id", "game_id", "liked", "rating"])
        writer.writerow([data.user_id, data.game_id, data.liked, data.rating])

# === API Endpoints ===

@app.post("/newInteraction")
def new_interaction(data: InteractionData):
    try:
        log_interaction(data)
        return {"success": True, "message": "Interaction recorded", "data": data.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cold-start")
def cold_start():
    recs = cold_start_recommendations(games_df, top_n=12)
    return recs.to_dict(orient="records")

@app.get("/recommend/tfidf")
def recommend_tfidf(user_id: str = "user1"):
    interactions = get_sample_interactions()
    recs = recommend_for_user(user_id, interactions, games_df, tfidf_matrix, top_n=12)
    return recs.to_dict(orient="records")

@app.get("/recommend/bert")
def recommend_bert(user_id: str = "user1"):
    interactions = get_sample_interactions()
    recs = recommend_for_user_sentence_transformer(user_id, interactions, games_df, embeddings, top_n=12)
    return recs.to_dict(orient="records")

@app.get("/recommend/hybrid-bert")
def recommend_hybrid_bert(user_id: str = "user1"):
    interactions = get_sample_interactions()
    recs = hybrid_recommendation_sentence_transformer(user_id, interactions, games_df, embeddings, top_n=12)
    return recs.to_dict(orient="records")

@app.get("/recommend/collaborative")
def recommend_collaborative(user_id: str = "user1"):
    interactions = get_sample_interactions()
    scores = get_collaborative_scores(user_id, interactions, games_df)
    games_with_scores = games_df.copy()
    games_with_scores["score"] = scores
    recs = games_with_scores.sort_values(by="score", ascending=False)
    return recs[["id", "name", "genre_text", "score", "background_image"]].head(12).to_dict(orient="records")

@app.get("/recommend/hybrid-tfidf")
def recommend_hybrid_tfidf(user_id: str = "user1"):
    interactions = get_sample_interactions()
    recs = hybrid_recommendation(user_id, interactions, games_df, tfidf_matrix, top_n=12)
    return recs.to_dict(orient="records")
