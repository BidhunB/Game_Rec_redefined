
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import ast
from tqdm import tqdm
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import glob

# === 1. DATA LOADING ===
import pandas as pd
import ast

import pickle
import os

def save_embeddings(embeddings, path="saved/embeddings.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)

def load_saved_embeddings(path="saved/embeddings.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def extract_names_from_column(col):
    
    def safe_extract(item):
        try:
            parsed = ast.literal_eval(item)
            if isinstance(parsed, list):
                return " ".join(obj["name"] for obj in parsed if isinstance(obj, dict) and "name" in obj)
        except (ValueError, SyntaxError):
            pass
        return str(item)
    
    return col.fillna("[]").apply(safe_extract)


def load_games_dataset(csv_path: str, nrows: int = 100):
    
    df = pd.read_csv(csv_path)
    df = df.sample(n=100, random_state=42)  # Randomly select 100 rows

    # Safely extract string features from complex columns
    df['genre_text'] = extract_names_from_column(df.get('genres', pd.Series()))
    df['tag_text'] = extract_names_from_column(df.get('tags', pd.Series()))

    # Normalize and lowercase basic columns
    for col in ['name', 'description']:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.lower()
        else:
            df[col] = ""

    # Combine all meaningful text fields into a single string for embedding
    df['combined'] = (
        df['name'] + " " +
        df['genre_text'] + " " +
        df['tag_text']
    ).str.lower()

    df = df.dropna(subset=['combined'])
    df.reset_index(drop=True, inplace=True)
    return df


def get_sample_interactions():
    
    print("Warning: get_sample_interactions() is deprecated. Use database.get_interactions_dataframe() instead.")
    return pd.DataFrame(columns=["user_id", "game_id", "liked", "rating"])

# === 2. FEATURE PREPARATION ===
def prepare_tfidf_matrix(games_df):
    
    vectorizer = TfidfVectorizer(stop_words='english')
    return vectorizer.fit_transform(games_df["combined"])


def sentence_transformer_model(games_df, batch_size=1, chunk_size=10):
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = games_df["combined"].tolist()
    embeddings = []
    embeddings = model.encode(texts, show_progress_bar=True)
    save_embeddings(embeddings)

    return np.array(embeddings)

# === 3. COLD START RECOMMENDATION ===
def cold_start_recommendations(games_df, top_n=10):
    
    sort_cols = [col for col in ['rating', 'ratings_count'] if col in games_df.columns]
    if not sort_cols:
        return games_df.sample(n=top_n)
    return games_df.sort_values(by=sort_cols, ascending=False).head(top_n)[["id", "name", "genre_text", "background_image"] + sort_cols]

# === 4. CONTENT-BASED RECOMMENDATION ===
def recommend_for_user(user_id, interactions_df, games_df, tfidf_matrix, top_n=10):
    
    # Step 1: Filter user's interactions
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    if user_interactions.empty:
        return pd.DataFrame()

    # Step 2: Match game IDs to DataFrame indices
    game_id_to_index = pd.Series(games_df.index.values, index=games_df['id']).dropna()
    rated_game_ids = user_interactions['game_id']
    rated_game_ids_in_games = rated_game_ids[rated_game_ids.isin(game_id_to_index.index)]
    valid_indices = game_id_to_index[rated_game_ids_in_games].astype(int).tolist()
    ratings = user_interactions[user_interactions['game_id'].isin(rated_game_ids_in_games)]['rating']

    if not valid_indices:
        return pd.DataFrame()

    # Step 4: Build user profile vector using weighted sum of TF-IDF vectors
    user_vector = np.zeros((tfidf_matrix.shape[1],))
    for idx, rating in zip(valid_indices, ratings):
        user_vector += np.asarray(tfidf_matrix[idx].toarray(), dtype=np.float64).flatten() * rating

    # Step 5: Normalize user vector
    if np.linalg.norm(user_vector) == 0:
        return pd.DataFrame()
    user_vector = normalize(user_vector.reshape(1, -1))

    # Step 6: Compute cosine similarity
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Step 7: Filter out already rated games
    games_df = games_df.copy()
    games_df['similarity'] = similarity_scores
    already_rated_ids = set(rated_game_ids)
    recs = games_df[~games_df['id'].isin(already_rated_ids)].sort_values(by='similarity', ascending=False)

    # Step 8: Return top recommendations
    return recs[['id', 'name', 'genre_text', 'similarity', 'background_image']].head(top_n)

# === 5. CONTENT-BASED USING BERT (SentenceTransformer) ===
def recommend_for_user_sentence_transformer(user_id, interactions_df, games_df, embeddings, top_n=10):
    
    # Filter user interactions
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]

    if user_interactions.empty:
        return pd.DataFrame()

    # Create a map from game ID to its index in embeddings
    game_id_to_index = pd.Series(games_df.index, index=games_df['id'])

    # Get ratings and corresponding embedding indices
    rated_game_ids = user_interactions['game_id']
    ratings = user_interactions['rating']

    valid_indices = []
    weights = []

    for game_id, rating in zip(rated_game_ids, ratings):
        if game_id in game_id_to_index:
            idx = game_id_to_index[game_id]
            valid_indices.append(idx)
            weights.append(rating)

    if not valid_indices:
        return pd.DataFrame()

    # Compute weighted average user vector
    user_vector = np.average(
        embeddings[valid_indices], 
        axis=0, 
        weights=weights
    ).reshape(1, -1)

    # Normalize for cosine similarity
    user_vector = normalize(user_vector)

    # Calculate similarity scores
    similarity_scores = cosine_similarity(user_vector, embeddings).flatten()

    # Exclude already rated games
    already_rated_ids = rated_game_ids.tolist()
    games_df = games_df.copy()
    games_df['similarity'] = similarity_scores

    recs = games_df[~games_df['id'].isin(already_rated_ids)]
    recs = recs.sort_values(by='similarity', ascending=False)

    return recs[['id', 'name', 'genre_text', 'similarity', 'background_image']].head(top_n)

# === 6. COLLABORATIVE FILTERING ===
def get_collaborative_scores(user_id, interactions_df, games_df):
    
    matrix = pd.pivot_table(
        interactions_df,
        values='liked',
        index='user_id',
        columns='game_id',
        fill_value=0
    ).astype(int)

    if user_id not in matrix.index:
        return np.zeros(games_df.shape[0])

    user_sim = cosine_similarity(matrix.loc[user_id].values.reshape(1, -1), matrix)[0]
    weighted_scores = np.dot(user_sim, matrix.values)
    normalized_scores = weighted_scores / (user_sim.sum() + 1e-9)

    # Vectorized mapping of game IDs to collaborative scores
    game_id_array = games_df['id'].to_numpy()
    matrix_columns = matrix.columns.to_numpy()
    idx_map = np.isin(game_id_array, matrix_columns)
    col_index = np.where(idx_map)[0]

    score_map = {gid: score for gid, score in zip(matrix.columns, normalized_scores)}
    scores = np.array([score_map.get(gid, 0.0) for gid in game_id_array])

    return scores

# === 7. HYBRID RECOMMENDATIONS ===
def get_content_scores(user_id, interactions_df, games_df, tfidf_matrix):
    
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    liked_ids = user_interactions[user_interactions['liked']]['game_id']
    disliked_ids = user_interactions[~user_interactions['liked']]['game_id']

    liked_indices = games_df[games_df['id'].isin(liked_ids)].index
    disliked_indices = games_df[games_df['id'].isin(disliked_ids)].index

    if liked_indices.empty and disliked_indices.empty:
        return np.zeros(tfidf_matrix.shape[0])

    # Ensure TF-IDF is a numpy array
    tfidf = tfidf_matrix.toarray() if hasattr(tfidf_matrix, 'toarray') else tfidf_matrix

    user_vector = np.zeros((1, tfidf.shape[1]))

    if not liked_indices.empty:
        user_vector += np.mean(tfidf[liked_indices], axis=0, keepdims=True)
    if not disliked_indices.empty:
        user_vector -= np.mean(tfidf[disliked_indices], axis=0, keepdims=True)

    user_vector = normalize(user_vector)

    return cosine_similarity(user_vector, tfidf).flatten()

def hybrid_recommendation(user_id, interactions_df, games_df, tfidf_matrix, top_n=10, content_weight=0.5, collab_weight=0.5):
    
    # Get scores
    content_scores = get_content_scores(user_id, interactions_df, games_df, tfidf_matrix)
    collab_scores = get_collaborative_scores(user_id, interactions_df, games_df)

    # Handle shape mismatches
    if content_scores.shape[0] != games_df.shape[0]:
        raise ValueError("Mismatch between content scores and number of games")
    if collab_scores.shape[0] != games_df.shape[0]:
        raise ValueError("Mismatch between collaborative scores and number of games")

    # Normalize scores if necessary (optional but recommended)
    def safe_normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9) if np.max(x) > np.min(x) else x

    content_scores = safe_normalize(content_scores)
    collab_scores = safe_normalize(collab_scores)

    # Combine weighted scores
    hybrid_scores = content_weight * content_scores + collab_weight * collab_scores

    # Prepare recommendations
    seen_ids = interactions_df[interactions_df['user_id'] == user_id]['game_id'].tolist()
    games_df = games_df.copy()
    games_df['score'] = hybrid_scores
    recs = games_df[~games_df['id'].isin(seen_ids)].sort_values(by='score', ascending=False)

    return recs[['id', 'name', 'genre_text', 'score', 'background_image']].head(top_n)

# === 8. HYBRID BERT RECOMMENDATIONS ===
def get_content_scores_sentence_transformer(user_id, interactions_df, games_df, embeddings):
    
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    liked_ids = user_interactions[user_interactions['liked']]['game_id']
    disliked_ids = user_interactions[~user_interactions['liked']]['game_id']

    liked_indices = games_df[games_df['id'].isin(liked_ids)].index
    disliked_indices = games_df[games_df['id'].isin(disliked_ids)].index

    if liked_indices.empty and disliked_indices.empty:
        return np.zeros(embeddings.shape[0])

    user_vector = np.zeros((1, embeddings.shape[1]))
    if not liked_indices.empty:
        user_vector += np.mean(embeddings[liked_indices], axis=0, keepdims=True)
    if not disliked_indices.empty:
        user_vector -= np.mean(embeddings[disliked_indices], axis=0, keepdims=True)

    return cosine_similarity(user_vector, embeddings).flatten()


def hybrid_recommendation_sentence_transformer(user_id, interactions_df, games_df, embeddings, top_n=10, content_weight=0.5, collab_weight=0.5):
    
    def safe_normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9) if np.max(x) > np.min(x) else x

    content_scores = get_content_scores_sentence_transformer(user_id, interactions_df, games_df, embeddings)
    collab_scores = get_collaborative_scores(user_id, interactions_df, games_df)

    content_scores = safe_normalize(content_scores)
    collab_scores = safe_normalize(collab_scores)

    hybrid_scores = content_weight * content_scores + collab_weight * collab_scores

    seen_ids = interactions_df[interactions_df['user_id'] == user_id]['game_id'].tolist()
    games_df = games_df.copy()
    games_df['score'] = hybrid_scores
    recs = games_df[~games_df['id'].isin(seen_ids)].sort_values(by='score', ascending=False)
    return recs[['id', 'name', 'genre_text', 'score', 'background_image']].head(top_n)
