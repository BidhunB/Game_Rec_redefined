# Game Recommendation System

This project is a simple content-based game recommender using TF-IDF and SentenceTransformer embeddings. It provides cold start and personalized recommendations based on user interactions.

## Features
- Cold start recommendations (popular games)
- Personalized recommendations using TF-IDF
- Personalized recommendations using SentenceTransformer embeddings
- Takes into account both liked and disliked games

## Dataset Download

This recommender requires a RAWG games dataset in CSV format. You can obtain the dataset from the [RAWG Video Games Database API](https://rawg.io/apidocs) or export it from the RAWG website.

- **RAWG API Docs:** https://rawg.io/apidocs
- You may need to register for an API key.
- Download or export the games data as a CSV file with at least the following columns: `id`, `name`, and either `genre_text` or `genres`.
- Save the file as `rawg_games.csv` in the project directory (or update the filename in `recommender_test.py`).

## Setup

1. **Clone the repository** (if needed):
   ```sh
   git clone <your-repo-url>
   cd game_rec
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Prepare your data:**
   - Place your RAWG games dataset CSV file in the project directory.
   - The file should be named `rawg_games.csv` by default, or you can change the filename in `recommender_test.py`.
   - The CSV should contain at least an `id`, `name`, and either a `genre_text` or `genres` column.

## Usage

Run the recommender test script:

```sh
python recommender_test.py
```

This will print:
- Cold start recommendations
- Personalized recommendations for a sample user using both TF-IDF and SentenceTransformer methods

## Customization
- To test with your own user interactions, edit the `get_sample_interactions()` function in `recommender_test.py`.
- To use a different CSV file, change the filename in the `run_test_program()` call at the bottom of `recommender_test.py`.

## Notes
- The first run may take longer as the SentenceTransformer model downloads.
- Make sure your CSV data matches the expected format for best results.

---

Enjoy experimenting with game recommendations! 