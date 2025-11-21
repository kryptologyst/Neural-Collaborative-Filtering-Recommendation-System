"""Utility functions for Neural Collaborative Filtering project."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cpu", "cuda").
        
    Returns:
        PyTorch device object.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device_obj = torch.device(device)
    logger.info(f"Using device: {device_obj}")
    return device_obj


def create_user_item_mapping(
    interactions: np.ndarray,
    user_col: str = "user_id",
    item_col: str = "item_id"
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """Create mappings between original IDs and internal indices.
    
    Args:
        interactions: Array of interactions with user and item columns.
        user_col: Name of the user column.
        item_col: Name of the item column.
        
    Returns:
        Tuple of (user_to_idx, idx_to_user, item_to_idx, idx_to_item) mappings.
    """
    unique_users = sorted(interactions[user_col].unique())
    unique_items = sorted(interactions[item_col].unique())
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    
    logger.info(f"Created mappings for {len(unique_users)} users and {len(unique_items)} items")
    
    return user_to_idx, idx_to_user, item_to_idx, idx_to_item


def calculate_sparsity(interactions: np.ndarray, n_users: int, n_items: int) -> float:
    """Calculate the sparsity of the interaction matrix.
    
    Args:
        interactions: Array of interactions.
        n_users: Number of users.
        n_items: Number of items.
        
    Returns:
        Sparsity value (0 = dense, 1 = sparse).
    """
    total_possible = n_users * n_items
    actual_interactions = len(interactions)
    sparsity = 1 - (actual_interactions / total_possible)
    return sparsity


def sample_negative_interactions(
    interactions: np.ndarray,
    n_users: int,
    n_items: int,
    ratio: float = 4.0,
    user_col: str = "user_id",
    item_col: str = "item_id"
) -> np.ndarray:
    """Sample negative interactions for training.
    
    Args:
        interactions: Positive interactions.
        n_users: Number of users.
        n_items: Number of items.
        ratio: Ratio of negative to positive samples.
        user_col: Name of the user column.
        item_col: Name of the item column.
        
    Returns:
        Array of negative interactions.
    """
    # Create set of positive interactions for fast lookup
    positive_interactions = set(
        (row[user_col], row[item_col]) for row in interactions
    )
    
    n_negative = int(len(interactions) * ratio)
    negative_interactions = []
    
    while len(negative_interactions) < n_negative:
        user_idx = np.random.randint(0, n_users)
        item_idx = np.random.randint(0, n_items)
        
        if (user_idx, item_idx) not in positive_interactions:
            negative_interactions.append([user_idx, item_idx, 0])
    
    negative_array = np.array(negative_interactions)
    logger.info(f"Sampled {len(negative_array)} negative interactions")
    
    return negative_array


def split_interactions_temporal(
    interactions: np.ndarray,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    timestamp_col: str = "timestamp"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split interactions temporally for evaluation.
    
    Args:
        interactions: Array of interactions sorted by timestamp.
        test_ratio: Ratio of interactions for testing.
        val_ratio: Ratio of interactions for validation.
        timestamp_col: Name of the timestamp column.
        
    Returns:
        Tuple of (train, validation, test) interaction arrays.
    """
    # Sort by timestamp
    sorted_indices = np.argsort(interactions[timestamp_col])
    sorted_interactions = interactions[sorted_indices]
    
    n_total = len(sorted_interactions)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_test - n_val
    
    train = sorted_interactions[:n_train]
    val = sorted_interactions[n_train:n_train + n_val]
    test = sorted_interactions[n_train + n_val:]
    
    logger.info(f"Split interactions: {len(train)} train, {len(val)} val, {len(test)} test")
    
    return train, val, test


def split_interactions_random(
    interactions: np.ndarray,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split interactions randomly for evaluation.
    
    Args:
        interactions: Array of interactions.
        test_ratio: Ratio of interactions for testing.
        val_ratio: Ratio of interactions for validation.
        random_state: Random state for reproducibility.
        
    Returns:
        Tuple of (train, validation, test) interaction arrays.
    """
    np.random.seed(random_state)
    n_total = len(interactions)
    indices = np.random.permutation(n_total)
    
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_test - n_val
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train = interactions[train_indices]
    val = interactions[val_indices]
    test = interactions[test_indices]
    
    logger.info(f"Split interactions: {len(train)} train, {len(val)} val, {len(test)} test")
    
    return train, val, test


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


class EarlyStopping:
    """Early stopping utility for training."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            mode: "min" for loss, "max" for metrics.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """Check if training should stop early.
        
        Args:
            score: Current score to evaluate.
            
        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
