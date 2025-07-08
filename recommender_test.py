import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# === 1. DATA LOADING ===
def load_games_dataset(csv_path: str):
    df = pd.read_csv(csv_path)

    # Create genre_text if missing
    if 'genre_text' not in df.columns:
        if 'genres' in df.columns:
            df['genre_text'] = df['genres'].fillna('[]').apply(
                lambda x: " ".join(eval(x)) if isinstance(x, str) and x.startswith('[') else str(x)
            )
        else:
            raise ValueError("CSV must contain either 'genre_text' or 'genres' column.")

    df = df.dropna(subset=["genre_text"])
    df["combined"] = df["genre_text"]
    df.reset_index(drop=True, inplace=True)
    return df

def get_sample_interactions():
    csv_path = "user_interactions.csv"
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        df["liked"] = df["liked"].astype(bool)
        df["rating"] = df["rating"].astype(int)
        return df
    else:
        print(f"File {csv_path} not found. Returning empty DataFrame.")
        return pd.DataFrame(columns=["user_id", "game_id", "liked", "rating"])

# === 2. FEATURE PREPARATION ===
def prepare_tfidf_matrix(games_df):
    vectorizer = TfidfVectorizer(stop_words='english')
    return vectorizer.fit_transform(games_df["combined"])

def sentence_transformer_model(games_df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(games_df["combined"].tolist(), show_progress_bar=True)

# === 3. COLD START RECOMMENDATION ===
def cold_start_recommendations(games_df, top_n=10):
    sort_cols = [col for col in ['rating', 'ratings_count'] if col in games_df.columns]
    if not sort_cols:
        return games_df.sample(n=top_n)
    return games_df.sort_values(by=sort_cols, ascending=False).head(top_n)[["id", "name", "genre_text", "background_image"] + sort_cols]

# === 4. CONTENT-BASED RECOMMENDATION ===
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
    return recs[['id', 'name', 'genre_text', 'similarity', 'background_image']].head(top_n)

# === 5. CONTENT-BASED USING BERT (SentenceTransformer) ===
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

    game_id_to_index = {gid: i for i, gid in enumerate(matrix.columns)}
    scores = np.zeros(games_df.shape[0])
    for i, gid in enumerate(games_df['id']):
        if gid in game_id_to_index:
            scores[i] = normalized_scores[game_id_to_index[gid]]
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

    user_vector = np.zeros((1, tfidf_matrix.shape[1]))
    if not liked_indices.empty:
        user_vector += tfidf_matrix[liked_indices].mean(axis=0)
    if not disliked_indices.empty:
        user_vector -= tfidf_matrix[disliked_indices].mean(axis=0)

    return cosine_similarity(user_vector, tfidf_matrix).flatten()

def hybrid_recommendation(user_id, interactions_df, games_df, tfidf_matrix, top_n=10, content_weight=0.5, collab_weight=0.5):
    content_scores = get_content_scores(user_id, interactions_df, games_df, tfidf_matrix)
    collab_scores = get_collaborative_scores(user_id, interactions_df, games_df)
    hybrid_scores = content_weight * content_scores + collab_weight * collab_scores

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
    content_scores = get_content_scores_sentence_transformer(user_id, interactions_df, games_df, embeddings)
    collab_scores = get_collaborative_scores(user_id, interactions_df, games_df)
    hybrid_scores = content_weight * content_scores + collab_weight * collab_scores

    seen_ids = interactions_df[interactions_df['user_id'] == user_id]['game_id'].tolist()
    games_df = games_df.copy()
    games_df['score'] = hybrid_scores
    recs = games_df[~games_df['id'].isin(seen_ids)].sort_values(by='score', ascending=False)
    return recs[['id', 'name', 'genre_text', 'score', 'background_image']].head(top_n)
