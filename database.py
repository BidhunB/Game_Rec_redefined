import os
import logging
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MongoDBService:
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self._users_collection: Optional[Collection] = None
        self._interactions_collection: Optional[Collection] = None

    def connect(self):
        """Establish connection to MongoDB and initialize collections."""
        try:
            mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
            db_name = os.getenv("MONGODB_DB", "game-recommender")
            self.client = MongoClient(mongodb_uri)
            self.db = self.client[db_name]
            self._users_collection = self.db["users"]
            self._interactions_collection = self.db["user_interactions"]
            self.create_indexes()
            logger.info("Connected to MongoDB successfully.")
        except Exception as e:
            logger.exception("Failed to connect to MongoDB.")
            raise

    def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB.")

    @property
    def interactions(self) -> Collection:
        if self._interactions_collection is None:
            raise RuntimeError("MongoDB not connected. Call connect() first.")
        return self._interactions_collection

    @property
    def users(self) -> Collection:
        if self._users_collection is None:
            raise RuntimeError("MongoDB not connected. Call connect() first.")
        return self._users_collection

    def create_indexes(self):
        """Create indexes to optimize query performance."""
        self.interactions.create_index([("user_id", 1), ("game_id", 1)])
        self.interactions.create_index([("user_id", 1), ("timestamp", -1)])

    def save_user_interaction(self, user_id: str, game_id: int, liked: bool, rating: int) -> bool:
        """Insert or update a user interaction."""
        try:
            interaction = {
                "user_id": user_id,
                "game_id": game_id,
                "liked": liked,
                "rating": rating,
                "timestamp": datetime.now()
            }
            self.interactions.update_one(
                {"user_id": user_id, "game_id": game_id},
                {"$set": interaction},
                upsert=True
            )
            return True
        except PyMongoError:
            logger.exception("Error saving interaction")
            return False

    def get_user_interactions(self, user_id: str) -> List[Dict[str, Any]]:
        """Fetch all interactions for a given user."""
        try:
            return list(self.interactions.find(
                {"user_id": user_id}, {"_id": 0}
            ))
        except PyMongoError:
            logger.exception("Error fetching user interactions")
            return []

    def get_all_interactions(self) -> List[Dict[str, Any]]:
        """Fetch all interactions in the database."""
        try:
            return list(self.interactions.find({}, {"_id": 0}))
        except PyMongoError:
            logger.exception("Error fetching all interactions")
            return []

    def get_interactions_dataframe(self) -> pd.DataFrame:
        """Return all interactions as a pandas DataFrame."""
        interactions = self.get_all_interactions()
        return self._convert_to_dataframe(interactions)

    def get_user_interactions_dataframe(self, user_id: str) -> pd.DataFrame:
        """Return interactions for a specific user as a pandas DataFrame."""
        interactions = self.get_user_interactions(user_id)
        return self._convert_to_dataframe(interactions)

    def _convert_to_dataframe(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Helper to convert interaction list to DataFrame."""
        if not data:
            return pd.DataFrame(columns=["user_id", "game_id", "liked", "rating", "timestamp"])
        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].astype(str)
        return df

    def delete_user_interaction(self, user_id: str, game_id: int) -> bool:
        """Delete a user interaction."""
        try:
            result = self.interactions.delete_one({
                "user_id": user_id,
                "game_id": game_id
            })
            return result.deleted_count > 0
        except PyMongoError:
            logger.exception("Error deleting interaction")
            return False

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Compute stats for a user."""
        try:
            pipeline = [
                {"$match": {"user_id": user_id}},
                {"$group": {
                    "_id": None,
                    "total_interactions": {"$sum": 1},
                    "liked_games": {"$sum": {"$cond": ["$liked", 1, 0]}},
                    "disliked_games": {"$sum": {"$cond": ["$liked", 0, 1]}},
                    "avg_rating": {"$avg": "$rating"},
                    "max_rating": {"$max": "$rating"},
                    "min_rating": {"$min": "$rating"},
                }}
            ]
            result = list(self.interactions.aggregate(pipeline))
            if result:
                stats = result[0]
                stats.pop("_id", None)
                return stats
            return self._default_stats()
        except PyMongoError:
            logger.exception("Error fetching user stats")
            return self._default_stats()

    def _default_stats(self) -> Dict[str, Any]:
        """Return default empty stats."""
        return {
            "total_interactions": 0,
            "liked_games": 0,
            "disliked_games": 0,
            "avg_rating": 0,
            "max_rating": 0,
            "min_rating": 0
        }

    def __del__(self):
        self.disconnect()

# Global instance (can be used as needed)
db_service = MongoDBService()
