"""Data loading and preprocessing utilities."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig

from ..utils import (
    calculate_sparsity,
    create_user_item_mapping,
    sample_negative_interactions,
    split_interactions_random,
    split_interactions_temporal,
)


class BaseDataLoader(ABC):
    """Abstract base class for data loaders."""
    
    def __init__(self, config: DictConfig):
        """Initialize data loader with configuration.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.interactions: Optional[pd.DataFrame] = None
        self.items: Optional[pd.DataFrame] = None
        self.users: Optional[pd.DataFrame] = None
        self.user_to_idx: Optional[Dict[int, int]] = None
        self.idx_to_user: Optional[Dict[int, int]] = None
        self.item_to_idx: Optional[Dict[int, int]] = None
        self.idx_to_item: Optional[Dict[int, int]] = None
        
    @abstractmethod
    def load_data(self) -> None:
        """Load the dataset."""
        pass
    
    def preprocess_data(self) -> None:
        """Preprocess the loaded data."""
        if self.interactions is None:
            raise ValueError("Data must be loaded before preprocessing")
        
        # Create mappings
        self.user_to_idx, self.idx_to_user, self.item_to_idx, self.idx_to_item = (
            create_user_item_mapping(self.interactions)
        )
        
        # Convert IDs to indices
        self.interactions["user_idx"] = self.interactions["user_id"].map(self.user_to_idx)
        self.interactions["item_idx"] = self.interactions["item_id"].map(self.item_to_idx)
        
        # Calculate statistics
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        sparsity = calculate_sparsity(self.interactions, n_users, n_items)
        
        logger.info(f"Dataset statistics:")
        logger.info(f"  Users: {n_users}")
        logger.info(f"  Items: {n_items}")
        logger.info(f"  Interactions: {len(self.interactions)}")
        logger.info(f"  Sparsity: {sparsity:.4f}")
    
    def get_train_val_test_split(
        self,
        temporal: bool = True,
        test_ratio: float = 0.1,
        val_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test sets.
        
        Args:
            temporal: Whether to use temporal splitting.
            test_ratio: Ratio of data for testing.
            val_ratio: Ratio of data for validation.
            
        Returns:
            Tuple of (train, validation, test) DataFrames.
        """
        if self.interactions is None:
            raise ValueError("Data must be loaded and preprocessed first")
        
        interactions_array = self.interactions[["user_idx", "item_idx", "rating", "timestamp"]].values
        
        if temporal:
            train, val, test = split_interactions_temporal(
                interactions_array, test_ratio, val_ratio
            )
        else:
            train, val, test = split_interactions_random(
                interactions_array, test_ratio, val_ratio
            )
        
        # Convert back to DataFrames
        columns = ["user_idx", "item_idx", "rating", "timestamp"]
        train_df = pd.DataFrame(train, columns=columns)
        val_df = pd.DataFrame(val, columns=columns)
        test_df = pd.DataFrame(test, columns=columns)
        
        return train_df, val_df, test_df
    
    def add_negative_samples(
        self,
        interactions: pd.DataFrame,
        ratio: float = 4.0
    ) -> pd.DataFrame:
        """Add negative samples to interactions.
        
        Args:
            interactions: Positive interactions.
            ratio: Ratio of negative to positive samples.
            
        Returns:
            DataFrame with positive and negative interactions.
        """
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        
        interactions_array = interactions[["user_idx", "item_idx", "rating"]].values
        negative_samples = sample_negative_interactions(
            interactions_array, n_users, n_items, ratio
        )
        
        # Combine positive and negative samples
        all_samples = np.vstack([interactions_array, negative_samples])
        
        # Convert back to DataFrame
        columns = ["user_idx", "item_idx", "rating"]
        result_df = pd.DataFrame(all_samples, columns=columns)
        
        logger.info(f"Added {len(negative_samples)} negative samples")
        
        return result_df


class SyntheticDataLoader(BaseDataLoader):
    """Synthetic data loader for testing and demonstration."""
    
    def __init__(self, config: DictConfig):
        """Initialize synthetic data loader.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
        self.n_users = config.n_users
        self.n_items = config.n_items
        self.n_interactions = config.n_interactions
        self.rating_scale = config.rating_scale
        self.sparsity = config.sparsity
        
    def load_data(self) -> None:
        """Generate synthetic interaction data."""
        logger.info("Generating synthetic dataset...")
        
        # Generate user and item IDs
        user_ids = list(range(self.n_users))
        item_ids = list(range(self.n_items))
        
        # Generate interactions based on sparsity
        total_possible = self.n_users * self.n_items
        n_interactions = int(total_possible * (1 - self.sparsity))
        
        # Sample user-item pairs
        user_item_pairs = []
        while len(user_item_pairs) < n_interactions:
            user_id = np.random.choice(user_ids)
            item_id = np.random.choice(item_ids)
            if (user_id, item_id) not in user_item_pairs:
                user_item_pairs.append((user_id, item_id))
        
        # Generate ratings with some structure
        ratings = []
        timestamps = []
        
        for user_id, item_id in user_item_pairs:
            # Add some user and item bias
            user_bias = np.random.normal(0, 0.5)
            item_bias = np.random.normal(0, 0.5)
            
            # Generate rating
            base_rating = 3.0 + user_bias + item_bias
            rating = np.clip(base_rating + np.random.normal(0, 0.5), 
                           self.rating_scale[0], self.rating_scale[1])
            rating = int(round(rating))
            
            # Generate timestamp (last 30 days)
            timestamp = np.random.uniform(0, 30 * 24 * 3600)  # seconds
            
            ratings.append(rating)
            timestamps.append(timestamp)
        
        # Create interactions DataFrame
        self.interactions = pd.DataFrame({
            "user_id": [pair[0] for pair in user_item_pairs],
            "item_id": [pair[1] for pair in user_item_pairs],
            "rating": ratings,
            "timestamp": timestamps
        })
        
        # Generate item metadata
        self.items = pd.DataFrame({
            "item_id": item_ids,
            "title": [f"Item {i}" for i in item_ids],
            "category": np.random.choice(range(10), self.n_items),
            "popularity": np.random.exponential(1.0, self.n_items)
        })
        
        # Generate user metadata
        self.users = pd.DataFrame({
            "user_id": user_ids,
            "age": np.random.randint(18, 65, self.n_users),
            "gender": np.random.choice(["M", "F"], self.n_users),
            "location": np.random.choice(["US", "EU", "AS"], self.n_users)
        })
        
        logger.info(f"Generated synthetic dataset with {len(self.interactions)} interactions")


class MovieLensDataLoader(BaseDataLoader):
    """MovieLens dataset loader."""
    
    def __init__(self, config: DictConfig):
        """Initialize MovieLens data loader.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__(config)
        self.data_path = config.data_path
        self.min_ratings = config.get("min_ratings", 5)
        
    def load_data(self) -> None:
        """Load MovieLens dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path {self.data_path} does not exist")
        
        logger.info(f"Loading MovieLens data from {self.data_path}")
        
        # Load interactions
        interactions_file = os.path.join(self.data_path, "ratings.csv")
        self.interactions = pd.read_csv(interactions_file)
        
        # Load items
        items_file = os.path.join(self.data_path, "movies.csv")
        if os.path.exists(items_file):
            self.items = pd.read_csv(items_file)
        
        # Load users
        users_file = os.path.join(self.data_path, "users.csv")
        if os.path.exists(users_file):
            self.users = pd.read_csv(users_file)
        
        # Filter users and items with minimum interactions
        if self.min_ratings > 0:
            user_counts = self.interactions["user_id"].value_counts()
            item_counts = self.interactions["item_id"].value_counts()
            
            valid_users = user_counts[user_counts >= self.min_ratings].index
            valid_items = item_counts[item_counts >= self.min_ratings].index
            
            self.interactions = self.interactions[
                (self.interactions["user_id"].isin(valid_users)) &
                (self.interactions["item_id"].isin(valid_items))
            ]
            
            logger.info(f"Filtered to {len(valid_users)} users and {len(valid_items)} items")


class RecommendationDataset:
    """PyTorch dataset for recommendation models."""
    
    def __init__(
        self,
        interactions: pd.DataFrame,
        include_negative: bool = False,
        negative_ratio: float = 4.0
    ):
        """Initialize recommendation dataset.
        
        Args:
            interactions: DataFrame with user_idx, item_idx, rating columns.
            include_negative: Whether to include negative samples.
            negative_ratio: Ratio of negative to positive samples.
        """
        self.interactions = interactions.copy()
        self.include_negative = include_negative
        
        if include_negative:
            # Add negative samples
            positive_interactions = self.interactions[self.interactions["rating"] > 0]
            negative_samples = self._generate_negative_samples(
                positive_interactions, negative_ratio
            )
            self.interactions = pd.concat([positive_interactions, negative_samples])
        
        logger.info(f"Dataset size: {len(self.interactions)}")
    
    def _generate_negative_samples(
        self,
        positive_interactions: pd.DataFrame,
        ratio: float
    ) -> pd.DataFrame:
        """Generate negative samples."""
        n_negative = int(len(positive_interactions) * ratio)
        
        # Get unique users and items
        unique_users = positive_interactions["user_idx"].unique()
        unique_items = positive_interactions["item_idx"].unique()
        
        # Create set of positive interactions for fast lookup
        positive_set = set(
            (row["user_idx"], row["item_idx"])
            for _, row in positive_interactions.iterrows()
        )
        
        negative_samples = []
        while len(negative_samples) < n_negative:
            user_idx = np.random.choice(unique_users)
            item_idx = np.random.choice(unique_items)
            
            if (user_idx, item_idx) not in positive_set:
                negative_samples.append({
                    "user_idx": user_idx,
                    "item_idx": item_idx,
                    "rating": 0
                })
        
        return pd.DataFrame(negative_samples)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.interactions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item at index."""
        row = self.interactions.iloc[idx]
        user_idx = torch.tensor(row["user_idx"], dtype=torch.long)
        item_idx = torch.tensor(row["item_idx"], dtype=torch.long)
        rating = torch.tensor(row["rating"], dtype=torch.float)
        
        return user_idx, item_idx, rating
