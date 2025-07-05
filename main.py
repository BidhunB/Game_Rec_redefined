from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from recommender_test import (
    load_games_dataset,
    cold_start_recommendations,
    prepare_tfidf_matrix,
    get_sample_interactions,
    recommend_for_user,
    sentence_transformer_model,
    recommend_for_user_sentence_transformer
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

@app.get("/cold-start")
def cold_start():
    recs = cold_start_recommendations(games_df, top_n=10)
    return recs.to_dict(orient="records")

@app.get("/recommend/tfidf")
def recommend_tfidf(user_id: str = "user1"):
    recs = recommend_for_user(user_id, interactions_df, games_df, tfidf_matrix, top_n=10)
    return recs.to_dict(orient="records")

@app.get("/recommend/bert")
def recommend_bert(user_id: str = "user1"):
    recs = recommend_for_user_sentence_transformer(user_id, interactions_df, games_df, embeddings, top_n=10)
    return recs.to_dict(orient="records")
