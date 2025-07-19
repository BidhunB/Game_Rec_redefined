from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
import threading
import uvicorn
import os

from recommender import (
    load_games_dataset,
    cold_start_recommendations,
    prepare_tfidf_matrix,
    recommend_for_user,
    sentence_transformer_model,
    recommend_for_user_sentence_transformer,
    hybrid_recommendation_sentence_transformer,
    get_collaborative_scores,
    hybrid_recommendation
)
from database import db_service

app = FastAPI()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render provides PORT env var
    uvicorn.run("main:app", host="0.0.0.0", port=port)


# === CORS Setup ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared variables
games_df = load_games_dataset("rawg_games.csv")
tfidf_matrix = prepare_tfidf_matrix(games_df)
embeddings = sentence_transformer_model(games_df)
lock = threading.Lock()

# Initialize database connection
db_service.connect()

# === Pydantic Models ===
class InteractionData(BaseModel):
    user_id: str
    game_id: int
    liked: bool
    rating: int
    timestamp: str

# === API Endpoints ===

@app.post("/newInteraction")
def new_interaction(data: InteractionData):
    try:
        # Sanitize and validate user_id
        user_id = data.user_id.strip()
        if not user_id or user_id in ["anonymous", "guest"]:
            raise HTTPException(status_code=400, detail="Invalid user ID")
        
        print(f"[NewInteraction] Saving interaction for user: {user_id}, game: {data.game_id}, liked: {data.liked}, rating: {data.rating}")
        
        success = db_service.save_user_interaction(
            user_id, 
            data.game_id, 
            data.liked, 
            data.rating
        )
        
        if success:
            print(f"[NewInteraction] Successfully saved interaction for user {user_id}")
            return {"success": True, "message": "Interaction recorded", "data": data.dict()}
        else:
            print(f"[NewInteraction] Failed to save interaction for user {user_id}")
            raise HTTPException(status_code=500, detail="Failed to save interaction")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[NewInteraction] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cold-start")
def cold_start():
    recs = cold_start_recommendations(games_df, top_n=12)
    return recs.to_dict(orient="records")

@app.get("/recommend/tfidf")
def recommend_tfidf(user_id: str = "user1"):
    print(f"[Recommend TF-IDF] User ID: {user_id}")
    
    # Sanitize user_id to prevent injection
    user_id = user_id.strip()
    if not user_id or user_id in ["anonymous", "guest"]:
        print(f"[Recommend TF-IDF] Invalid user ID: {user_id}, using cold start")
        recs = cold_start_recommendations(games_df, top_n=12)
        return recs.to_dict(orient="records")
    
    try:
        interactions = db_service.get_interactions_dataframe()
        print(f"[Recommend TF-IDF] Found {len(interactions)} total interactions")
        
        # Get user-specific interactions
        user_interactions = db_service.get_user_interactions_dataframe(user_id)
        print(f"[Recommend TF-IDF] User {user_id} has {len(user_interactions)} interactions")
        
        recs = recommend_for_user(user_id, interactions, games_df, tfidf_matrix, top_n=12)
        print(f"[Recommend TF-IDF] Generated {len(recs)} recommendations for user {user_id}")
        return recs.to_dict(orient="records")
    except Exception as e:
        print(f"[Recommend TF-IDF] Error: {e}")
        # Fallback to cold start
        recs = cold_start_recommendations(games_df, top_n=12)
        return recs.to_dict(orient="records")

@app.get("/recommend/bert")
def recommend_bert(user_id: str = "user1"):
    print(f"[Recommend BERT] User ID: {user_id}")
    
    # Sanitize user_id to prevent injection
    user_id = user_id.strip()
    if not user_id or user_id in ["anonymous", "guest"]:
        print(f"[Recommend BERT] Invalid user ID: {user_id}, using cold start")
        recs = cold_start_recommendations(games_df, top_n=12)
        return recs.to_dict(orient="records")
    
    try:
        interactions = db_service.get_interactions_dataframe()
        print(f"[Recommend BERT] Found {len(interactions)} total interactions")
        
        # Get user-specific interactions
        user_interactions = db_service.get_user_interactions_dataframe(user_id)
        print(f"[Recommend BERT] User {user_id} has {len(user_interactions)} interactions")
        
        recs = recommend_for_user_sentence_transformer(user_id, interactions, games_df, embeddings, top_n=12)
        print(f"[Recommend BERT] Generated {len(recs)} recommendations for user {user_id}")
        return recs.to_dict(orient="records")
    except Exception as e:
        print(f"[Recommend BERT] Error: {e}")
        # Fallback to cold start
        recs = cold_start_recommendations(games_df, top_n=12)
        return recs.to_dict(orient="records")

@app.get("/recommend/hybrid-bert")
def recommend_hybrid_bert(user_id: str = "user1"):
    print(f"[Recommend Hybrid BERT] User ID: {user_id}")
    
    # Sanitize user_id to prevent injection
    user_id = user_id.strip()
    if not user_id or user_id in ["anonymous", "guest"]:
        print(f"[Recommend Hybrid BERT] Invalid user ID: {user_id}, using cold start")
        recs = cold_start_recommendations(games_df, top_n=12)
        return recs.to_dict(orient="records")
    
    try:
        interactions = db_service.get_interactions_dataframe()
        print(f"[Recommend Hybrid BERT] Found {len(interactions)} total interactions")
        
        # Get user-specific interactions
        user_interactions = db_service.get_user_interactions_dataframe(user_id)
        print(f"[Recommend Hybrid BERT] User {user_id} has {len(user_interactions)} interactions")
        
        recs = hybrid_recommendation_sentence_transformer(user_id, interactions, games_df, embeddings, top_n=12)
        print(f"[Recommend Hybrid BERT] Generated {len(recs)} recommendations for user {user_id}")
        return recs.to_dict(orient="records")
    except Exception as e:
        print(f"[Recommend Hybrid BERT] Error: {e}")
        # Fallback to cold start
        recs = cold_start_recommendations(games_df, top_n=12)
        return recs.to_dict(orient="records")

@app.get("/recommend/collaborative")
def recommend_collaborative(user_id: str = "user1"):
    print(f"[Recommend Collaborative] User ID: {user_id}")
    
    # Sanitize user_id to prevent injection
    user_id = user_id.strip()
    if not user_id or user_id in ["anonymous", "guest"]:
        print(f"[Recommend Collaborative] Invalid user ID: {user_id}, using cold start")
        recs = cold_start_recommendations(games_df, top_n=12)
        return recs.to_dict(orient="records")
    
    try:
        interactions = db_service.get_interactions_dataframe()
        print(f"[Recommend Collaborative] Found {len(interactions)} total interactions")
        
        # Get user-specific interactions
        user_interactions = db_service.get_user_interactions_dataframe(user_id)
        print(f"[Recommend Collaborative] User {user_id} has {len(user_interactions)} interactions")
        
        scores = get_collaborative_scores(user_id, interactions, games_df)
        games_with_scores = games_df.copy()
        games_with_scores["score"] = scores
        recs = games_with_scores.sort_values(by="score", ascending=False)
        result = recs[["id", "name", "genre_text", "score", "background_image"]].head(12).to_dict(orient="records")
        print(f"[Recommend Collaborative] Generated {len(result)} recommendations for user {user_id}")
        return result
    except Exception as e:
        print(f"[Recommend Collaborative] Error: {e}")
        # Fallback to cold start
        recs = cold_start_recommendations(games_df, top_n=12)
        return recs.to_dict(orient="records")

@app.get("/recommend/hybrid-tfidf")
def recommend_hybrid_tfidf(user_id: str = "user1"):
    print(f"[Recommend Hybrid TF-IDF] User ID: {user_id}")
    
    # Sanitize user_id to prevent injection
    user_id = user_id.strip()
    if not user_id or user_id in ["anonymous", "guest"]:
        print(f"[Recommend Hybrid TF-IDF] Invalid user ID: {user_id}, using cold start")
        recs = cold_start_recommendations(games_df, top_n=12)
        return recs.to_dict(orient="records")
    
    try:
        interactions = db_service.get_interactions_dataframe()
        print(f"[Recommend Hybrid TF-IDF] Found {len(interactions)} total interactions")
        
        # Get user-specific interactions
        user_interactions = db_service.get_user_interactions_dataframe(user_id)
        print(f"[Recommend Hybrid TF-IDF] User {user_id} has {len(user_interactions)} interactions")
        
        recs = hybrid_recommendation(user_id, interactions, games_df, tfidf_matrix, top_n=12)
        print(f"[Recommend Hybrid TF-IDF] Generated {len(recs)} recommendations for user {user_id}")
        return recs.to_dict(orient="records")
    except Exception as e:
        print(f"[Recommend Hybrid TF-IDF] Error: {e}")
        # Fallback to cold start
        recs = cold_start_recommendations(games_df, top_n=12)
        return recs.to_dict(orient="records")

@app.get("/user/stats/{user_id}")
def get_user_stats(user_id: str):
    """Get user statistics"""
    try:
        stats = db_service.get_user_stats(user_id)
        return {"success": True, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/interactions/{user_id}")
def get_user_interactions(user_id: str):
    """Get user interactions"""
    try:
        interactions = db_service.get_user_interactions(user_id)
        return {"success": True, "interactions": interactions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))