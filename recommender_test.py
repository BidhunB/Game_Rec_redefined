import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# === 1. LOAD RAWG DATA FROM CSV ===

def load_games_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    
    # Create genre_text if it doesn't exist by combining genres or tags
    if 'genre_text' not in df.columns:
        if 'genres' in df.columns:
            # Try to convert genre list/dict into text
            df['genre_text'] = df['genres'].fillna('[]').apply(
                lambda x: " ".join(eval(x)) if isinstance(x, str) and x.startswith('[') else str(x)
            )
        else:
            raise ValueError("CSV must contain either 'genre_text' or 'genres' column.")
    
    # Drop rows where genre_text is missing or empty
    df = df.dropna(subset=["genre_text"])
    df["combined"] = df["genre_text"]  # You can expand with more features later
    df.reset_index(drop=True, inplace=True)
    return df


# === 2. COLD START RECOMMENDATION ===

def cold_start_recommendations(games_df, top_n=10):
    sort_cols = [col for col in ['rating', 'ratings_count'] if col in games_df.columns]
    if not sort_cols:
        return games_df.sample(n=top_n)
    return games_df.sort_values(by=sort_cols, ascending=False).head(top_n)[["id", "name", "genre_text"] + sort_cols]


# === 3. CONTENT-BASED RECOMMENDATION ===

def prepare_tfidf_matrix(games_df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(games_df["combined"])
    return tfidf_matrix

def sentence_transformer_model(games_df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(games_df["combined"].tolist(), show_progress_bar=True)
    return embeddings

def recommend_for_user(user_id, interactions_df, games_df, tfidf_matrix, top_n=10):
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    liked_ids = user_interactions[user_interactions['liked']]['game_id']
    disliked_ids = user_interactions[~user_interactions['liked']]['game_id']

    liked_indices = games_df[games_df['id'].isin(liked_ids)].index
    disliked_indices = games_df[games_df['id'].isin(disliked_ids)].index

    if liked_indices.empty and disliked_indices.empty:
        return pd.DataFrame()

    user_vector = np.zeros((1, tfidf_matrix.shape[1]))
    if not liked_indices.empty:
        user_vector += tfidf_matrix[liked_indices].mean(axis=0)
    if not disliked_indices.empty:
        user_vector -= tfidf_matrix[disliked_indices].mean(axis=0)

    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    already_seen = user_interactions['game_id'].tolist()
    games_df = games_df.copy()
    games_df['similarity'] = similarity_scores
    recs = games_df[~games_df['id'].isin(already_seen)].sort_values(by='similarity', ascending=False)
    return recs[['id', 'name', 'genre_text', 'similarity']].head(top_n)

def recommend_for_user_sentence_transformer(user_id, interactions_df, games_df, embeddings, top_n=10):
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    liked_ids = user_interactions[user_interactions['liked']]['game_id']
    disliked_ids = user_interactions[~user_interactions['liked']]['game_id']

    liked_indices = games_df[games_df['id'].isin(liked_ids)].index
    disliked_indices = games_df[games_df['id'].isin(disliked_ids)].index

    if liked_indices.empty and disliked_indices.empty:
        return pd.DataFrame()

    user_vector = np.zeros((1, embeddings.shape[1]))
    if not liked_indices.empty:
        user_vector += np.mean(embeddings[liked_indices], axis=0, keepdims=True)
    if not disliked_indices.empty:
        user_vector -= np.mean(embeddings[disliked_indices], axis=0, keepdims=True)

    similarity_scores = cosine_similarity(user_vector, embeddings).flatten()
    already_seen = user_interactions['game_id'].tolist()
    games_df = games_df.copy()
    games_df['similarity'] = similarity_scores
    recs = games_df[~games_df['id'].isin(already_seen)].sort_values(by='similarity', ascending=False)
    return recs[['id', 'name', 'genre_text', 'similarity']].head(top_n)

# === 4. SIMULATED USER INTERACTIONS ===

def get_sample_interactions():
    return pd.DataFrame([
        {"user_id": "user1", "game_id": 3498, "liked": True, "rating": 5},   # GTA V
        {"user_id": "user1", "game_id": 28, "liked": True, "rating": 5}, #red dead redemption 2
        {"user_id": "user1", "game_id": 58134, "liked": True, "rating": 4}, # marvels spiderman
        {"user_id": "user1", "game_id": 22509, "liked": True, "rating": 4}, # Minecraft
        {"user_id": "user1", "game_id": 42895, "liked": True, "rating": 4},# Assassin's Creed syndicate
        {"user_id": "user1", "game_id": 437059, "liked": False, "rating": 2}   # Assassin's Creed Valhalla
    ])


# === 5. RUN THE PROGRAM ===

def run_test_program(csv_path: str):
    games_df = load_games_dataset(csv_path)
    
    print("\n🎮 Cold Start Recommendations:")
    print(cold_start_recommendations(games_df, top_n=5).to_string(index=False))

    tfidf_matrix = prepare_tfidf_matrix(games_df)
    
    interactions_df = get_sample_interactions()

    print("\n🧠 Personalized Recommendations for user1 (TF-IDF):")
    personalized = recommend_for_user("user1", interactions_df, games_df, tfidf_matrix, top_n=10)
    if not personalized.empty:
        print(personalized.to_string(index=False))
    else:
        print("Not enough interaction data for personalized recommendations.")

    print("\n🤖 Personalized Recommendations for user1 (SentenceTransformer):")
    embeddings = sentence_transformer_model(games_df)
    personalized_st = recommend_for_user_sentence_transformer("user1", interactions_df, games_df, embeddings, top_n=10)
    if not personalized_st.empty:
        print(personalized_st.to_string(index=False))
    else:
        print("Not enough interaction data for personalized recommendations (SentenceTransformer).")


# === USAGE ===
# Replace 'games.csv' with your actual RAWG dataset filename
# Uncomment below line when you're ready to run
run_test_program("rawg_games.csv")
