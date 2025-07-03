import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


def recommend_for_user(user_id, interactions_df, games_df, tfidf_matrix, top_n=10):
    liked_game_ids = interactions_df[
        (interactions_df['user_id'] == user_id) & (interactions_df['liked'])
    ]['game_id']
    
    if liked_game_ids.empty:
        return pd.DataFrame()

    liked_indices = games_df[games_df['id'].isin(liked_game_ids)].index
    if liked_indices.empty:
        return pd.DataFrame()

    # FIX HERE — convert to array
    import numpy as np
    user_vector = np.asarray(tfidf_matrix[liked_indices].mean(axis=0))

    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    already_seen = interactions_df[interactions_df['user_id'] == user_id]['game_id'].tolist()
    games_df = games_df.copy()
    games_df['similarity'] = similarity_scores
    recs = games_df[~games_df['id'].isin(already_seen)].sort_values(by='similarity', ascending=False)

    return recs[['id', 'name', 'genre_text', 'similarity']].head(top_n)

# === 4. SIMULATED USER INTERACTIONS ===

def get_sample_interactions():
    return pd.DataFrame([
        {"user_id": "user1", "game_id": 3498, "liked": True, "rating": 5},   # GTA V
        {"user_id": "user1", "game_id": 4200, "liked": True, "rating": 4},   # Portal 2
        {"user_id": "user1", "game_id": 5286, "liked": False, "rating": 2}   # Disliked
    ])


# === 5. RUN THE PROGRAM ===

def run_test_program(csv_path: str):
    games_df = load_games_dataset(csv_path)
    
    print("\n🎮 Cold Start Recommendations:")
    print(cold_start_recommendations(games_df, top_n=5).to_string(index=False))

    tfidf_matrix = prepare_tfidf_matrix(games_df)
    interactions_df = get_sample_interactions()

    print("\n🧠 Personalized Recommendations for user1:")
    personalized = recommend_for_user("user1", interactions_df, games_df, tfidf_matrix, top_n=10)
    if not personalized.empty:
        print(personalized.to_string(index=False))
    else:
        print("Not enough interaction data for personalized recommendations.")


# === USAGE ===
# Replace 'games.csv' with your actual RAWG dataset filename
# Uncomment below line when you're ready to run
run_test_program("rawg_games.csv")
