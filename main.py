from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import json
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

# Allow requests from frontend (like Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load everything once when server starts
games_df = load_games_dataset("rawg_games.csv")
tfidf_matrix = prepare_tfidf_matrix(games_df)
embeddings = sentence_transformer_model(games_df)
interactions_df = get_sample_interactions()

# Pydantic model for like data
class LikeData(BaseModel):
    game_id: int
    game_name: str
    action: str  # "like" or "unlike"
    timestamp: str

@app.post("/like")
def like_game(like_data: LikeData):
    """
    Handle game likes/unlikes
    This endpoint receives like data and can be used to:
    - Store user preferences
    - Update recommendation models
    - Track user interactions
    """
    try:
        # Log the like action
        print(f"User {like_data.action} game: {like_data.game_name} (ID: {like_data.game_id}) at {like_data.timestamp}")
        
        # Here you could:
        # 1. Store in database
        # 2. Update user preferences
        # 3. Retrain recommendation models
        # 4. Send analytics
        
        # For now, just return success
        return {
            "success": True,
            "message": f"Successfully {like_data.action} {like_data.game_name}",
            "game_id": like_data.game_id,
            "action": like_data.action,
            "timestamp": like_data.timestamp
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/cold-start")
def cold_start():
    recs = cold_start_recommendations(games_df, top_n=12)
    return recs.to_dict(orient="records")

@app.get("/recommend/tfidf")
def recommend_tfidf(user_id: str = "user1"):
    recs = recommend_for_user(user_id, interactions_df, games_df, tfidf_matrix, top_n=12)
    return recs.to_dict(orient="records")

@app.get("/recommend/bert")
def recommend_bert(user_id: str = "user1"):
    recs = recommend_for_user_sentence_transformer(user_id, interactions_df, games_df, embeddings, top_n=12)
    return recs.to_dict(orient="records")

@app.get("/recommend/hybrid-bert")
def recommend_hybrid_bert(user_id: str = "user1"):
    recs = hybrid_recommendation_sentence_transformer(user_id, interactions_df, games_df, embeddings, top_n=12)
    return recs.to_dict(orient="records")

@app.get("/recommend/collaborative")
def recommend_collaborative(user_id: str = "user1"):
    scores = get_collaborative_scores(user_id, interactions_df, games_df)
    games_df_copy = games_df.copy()
    games_df_copy['score'] = scores
    recs = games_df_copy.sort_values(by='score', ascending=False)
    recs = recs[['id', 'name', 'genre_text', 'score', 'background_image']].head(12)
    return recs.to_dict(orient="records")

@app.get("/recommend/hybrid-tfidf")
def recommend_hybrid_tfidf(user_id: str = "user1"):
    recs = hybrid_recommendation(user_id, interactions_df, games_df, tfidf_matrix, top_n=12)
    return recs.to_dict(orient="records")
