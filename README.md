# Game Recommendation System

A full-stack, hybrid game recommendation platform that leverages content-based, collaborative, and deep learning models to provide personalized game suggestions. Includes a FastAPI backend and a modern Next.js frontend.

---

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset Requirements](#dataset-requirements)
- [Setup Instructions](#setup-instructions)
  - [Backend](#backend)
  - [Frontend](#frontend)
- [Usage](#usage)
- [Customization](#customization)
- [Contributing](#contributing)
- [Credits](#credits)

---

## Features
- **Cold Start Recommendations:** Popular games for new users
- **Personalized Recommendations:**
  - TF-IDF content-based filtering
  - SentenceTransformer (BERT) embeddings
  - Collaborative filtering
  - Hybrid models (content + collaborative)
- **User Feedback:** Like/dislike and rating support
- **Modern Web UI:** Built with Next.js and React
- **Automatic Data Refresh:** Backend reloads game data periodically

---

## Tech Stack
- **Backend:** Python, FastAPI, Pandas, scikit-learn, APScheduler, SentenceTransformers
- **Frontend:** Next.js (React, TypeScript, TailwindCSS)

---

## Dataset Requirements
- **Game Data:** CSV file (default: `rawg_games.csv`) with at least these columns:
  - `id`, `name`, and either `genre_text` or `genres`
  - Optional: `rating`, `ratings_count`, `background_image`
- **User Interactions:** CSV file (`user_interactions.csv`) with columns:
  - `user_id`, `game_id`, `liked`, `rating`
- Example game data can be downloaded from the [RAWG Video Games Database API](https://rawg.io/apidocs).

---

## Setup Instructions

### Backend
1. **Install Python dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Prepare your datasets:**
   - Place your main games CSV as `rawg_games.csv` in the project root.
   - (Optional) Place `user_interactions.csv` in the root for custom user data.
3. **Run the backend server:**
   ```sh
   python -m uvicorn main:app --reload
   ```
   The FastAPI server will start (default: http://127.0.0.1:8000).

### Frontend
1. **Install Node.js dependencies:**
   ```sh
   cd game-recommender-frontend
   npm install
   ```
2. **Start the development server:**
   ```sh
   npm run dev
   ```
   The app will be available at [http://localhost:3000](http://localhost:3000).

---

## Usage
- **API Endpoints:**
  - `POST /newInteraction` — Log a user interaction
  - `GET /cold-start` — Get cold start recommendations
  - `GET /recommend/tfidf?user_id=...` — TF-IDF recommendations
  - `GET /recommend/bert?user_id=...` — BERT-based recommendations
  - `GET /recommend/collaborative?user_id=...` — Collaborative filtering
  - `GET /recommend/hybrid-tfidf?user_id=...` — Hybrid (TF-IDF + collaborative)
  - `GET /recommend/hybrid-bert?user_id=...` — Hybrid (BERT + collaborative)
- **Frontend:**
  - Interact with the web UI to get recommendations, like/dislike games, and view stats.

---

## Customization
- **Change dataset:** Edit the filename in `main.py` and `recommender_test.py` if needed.
- **User interactions:** The backend reads/writes to `user_interactions.csv`.
- **Frontend:** Modify or extend components in `game-recommender-frontend/src/components`.

---

## Contributing
Contributions are welcome! Please open issues or pull requests for improvements.

---

## Credits
**Developed by:**
- [Joel Joy](https://github.com/Joeljoy1237)
- [Bidhun B](https://github.com/BidhunB/)
- [Varghese Francis](https://github.com/VargheeseFrancis)
- [Ashok Xavier](https://github.com/AshokXavier)

Special thanks to the open-source community and the RAWG API.

---

Enjoy experimenting with game recommendations! 
