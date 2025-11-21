"""Evaluation metrics and utilities for recommendation systems."""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig


class RankingMetrics:
    """Ranking-based evaluation metrics."""
    
    @staticmethod
    def precision_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """Calculate Precision@K.
        
        Args:
            recommended_items: List of recommended item indices.
            relevant_items: List of relevant item indices.
            k: Number of top recommendations to consider.
            
        Returns:
            Precision@K value.
        """
        if k == 0:
            return 0.0
        
        top_k = recommended_items[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant_items))
        
        return relevant_in_top_k / k
    
    @staticmethod
    def recall_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """Calculate Recall@K.
        
        Args:
            recommended_items: List of recommended item indices.
            relevant_items: List of relevant item indices.
            k: Number of top recommendations to consider.
            
        Returns:
            Recall@K value.
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k = recommended_items[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant_items))
        
        return relevant_in_top_k / len(relevant_items)
    
    @staticmethod
    def ndcg_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """Calculate NDCG@K.
        
        Args:
            recommended_items: List of recommended item indices.
            relevant_items: List of relevant item indices.
            k: Number of top recommendations to consider.
            
        Returns:
            NDCG@K value.
        """
        if k == 0 or len(relevant_items) == 0:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_items:
                dcg += 1.0 / math.log2(i + 2)
        
        # Calculate IDCG
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def map_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """Calculate MAP@K.
        
        Args:
            recommended_items: List of recommended item indices.
            relevant_items: List of relevant item indices.
            k: Number of top recommendations to consider.
            
        Returns:
            MAP@K value.
        """
        if k == 0 or len(relevant_items) == 0:
            return 0.0
        
        top_k = recommended_items[:k]
        relevant_in_top_k = [item for item in top_k if item in relevant_items]
        
        if not relevant_in_top_k:
            return 0.0
        
        # Calculate average precision
        precision_sum = 0.0
        for i, item in enumerate(relevant_in_top_k):
            rank = top_k.index(item) + 1
            precision_at_rank = (i + 1) / rank
            precision_sum += precision_at_rank
        
        return precision_sum / len(relevant_items)
    
    @staticmethod
    def hit_rate_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """Calculate Hit Rate@K.
        
        Args:
            recommended_items: List of recommended item indices.
            relevant_items: List of relevant item indices.
            k: Number of top recommendations to consider.
            
        Returns:
            Hit Rate@K value (0 or 1).
        """
        if k == 0:
            return 0.0
        
        top_k = recommended_items[:k]
        return 1.0 if len(set(top_k) & set(relevant_items)) > 0 else 0.0
    
    @staticmethod
    def mrr_at_k(recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """Calculate MRR@K.
        
        Args:
            recommended_items: List of recommended item indices.
            relevant_items: List of relevant item indices.
            k: Number of top recommendations to consider.
            
        Returns:
            MRR@K value.
        """
        if k == 0 or len(relevant_items) == 0:
            return 0.0
        
        top_k = recommended_items[:k]
        for i, item in enumerate(top_k):
            if item in relevant_items:
                return 1.0 / (i + 1)
        
        return 0.0


class DiversityMetrics:
    """Diversity-based evaluation metrics."""
    
    @staticmethod
    def intra_list_diversity(recommended_items: List[int], item_features: Optional[np.ndarray] = None) -> float:
        """Calculate intra-list diversity.
        
        Args:
            recommended_items: List of recommended item indices.
            item_features: Item feature matrix.
            
        Returns:
            Intra-list diversity value.
        """
        if len(recommended_items) <= 1:
            return 0.0
        
        if item_features is not None:
            # Use feature-based diversity
            features = item_features[recommended_items]
            distances = []
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    dist = np.linalg.norm(features[i] - features[j])
                    distances.append(dist)
            return np.mean(distances) if distances else 0.0
        else:
            # Use simple diversity (number of unique items)
            return len(set(recommended_items)) / len(recommended_items)
    
    @staticmethod
    def coverage(recommended_items: List[int], all_items: List[int]) -> float:
        """Calculate catalog coverage.
        
        Args:
            recommended_items: List of recommended item indices.
            all_items: List of all available item indices.
            
        Returns:
            Coverage value.
        """
        if len(all_items) == 0:
            return 0.0
        
        unique_recommended = set(recommended_items)
        return len(unique_recommended) / len(all_items)
    
    @staticmethod
    def novelty(recommended_items: List[int], item_popularity: Optional[np.ndarray] = None) -> float:
        """Calculate novelty of recommendations.
        
        Args:
            recommended_items: List of recommended item indices.
            item_popularity: Array of item popularity scores.
            
        Returns:
            Novelty value.
        """
        if item_popularity is not None:
            # Use popularity-based novelty
            popularity_scores = item_popularity[recommended_items]
            return -np.mean(np.log(popularity_scores + 1e-8))
        else:
            # Simple novelty (inverse of frequency)
            item_counts = {}
            for item in recommended_items:
                item_counts[item] = item_counts.get(item, 0) + 1
            
            novelty_scores = [1.0 / count for count in item_counts.values()]
            return np.mean(novelty_scores)


class RecommendationEvaluator:
    """Main evaluation class for recommendation models."""
    
    def __init__(self, config: DictConfig):
        """Initialize evaluator.
        
        Args:
            config: Evaluation configuration.
        """
        self.config = config
        self.metrics = config.metrics
        self.k_values = config.k_values
        self.settings = config.settings
        
        self.ranking_metrics = RankingMetrics()
        self.diversity_metrics = DiversityMetrics()
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        test_data: np.ndarray,
        user_item_matrix: Optional[np.ndarray] = None,
        item_features: Optional[np.ndarray] = None,
        item_popularity: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate a recommendation model.
        
        Args:
            model: Trained recommendation model.
            test_data: Test interactions.
            user_item_matrix: User-item interaction matrix.
            item_features: Item feature matrix.
            item_popularity: Item popularity scores.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        model.eval()
        results = {}
        
        # Get unique users for evaluation
        unique_users = np.unique(test_data[:, 0])
        if self.settings.get("sample_users"):
            n_sample = self.settings["sample_users"]
            unique_users = np.random.choice(unique_users, min(n_sample, len(unique_users)), replace=False)
        
        logger.info(f"Evaluating on {len(unique_users)} users")
        
        # Evaluate ranking metrics
        if "ranking" in self.metrics:
            ranking_results = self._evaluate_ranking_metrics(
                model, test_data, unique_users, user_item_matrix
            )
            results.update(ranking_results)
        
        # Evaluate diversity metrics
        if "diversity" in self.metrics:
            diversity_results = self._evaluate_diversity_metrics(
                model, test_data, unique_users, item_features, item_popularity
            )
            results.update(diversity_results)
        
        return results
    
    def _evaluate_ranking_metrics(
        self,
        model: torch.nn.Module,
        test_data: np.ndarray,
        unique_users: np.ndarray,
        user_item_matrix: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate ranking metrics."""
        results = {}
        
        for k in self.k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            map_scores = []
            hit_rate_scores = []
            mrr_scores = []
            
            for user_idx in unique_users:
                # Get relevant items for this user
                user_interactions = test_data[test_data[:, 0] == user_idx]
                relevant_items = user_interactions[:, 1].tolist()
                
                if len(relevant_items) == 0:
                    continue
                
                # Get recommendations for this user
                recommended_items = self._get_recommendations(model, user_idx, k * 2)
                
                # Calculate metrics
                precision_scores.append(
                    self.ranking_metrics.precision_at_k(recommended_items, relevant_items, k)
                )
                recall_scores.append(
                    self.ranking_metrics.recall_at_k(recommended_items, relevant_items, k)
                )
                ndcg_scores.append(
                    self.ranking_metrics.ndcg_at_k(recommended_items, relevant_items, k)
                )
                map_scores.append(
                    self.ranking_metrics.map_at_k(recommended_items, relevant_items, k)
                )
                hit_rate_scores.append(
                    self.ranking_metrics.hit_rate_at_k(recommended_items, relevant_items, k)
                )
                mrr_scores.append(
                    self.ranking_metrics.mrr_at_k(recommended_items, relevant_items, k)
                )
            
            # Average metrics across users
            results[f"precision@{k}"] = np.mean(precision_scores)
            results[f"recall@{k}"] = np.mean(recall_scores)
            results[f"ndcg@{k}"] = np.mean(ndcg_scores)
            results[f"map@{k}"] = np.mean(map_scores)
            results[f"hit_rate@{k}"] = np.mean(hit_rate_scores)
            results[f"mrr@{k}"] = np.mean(mrr_scores)
        
        return results
    
    def _evaluate_diversity_metrics(
        self,
        model: torch.nn.Module,
        test_data: np.ndarray,
        unique_users: np.ndarray,
        item_features: Optional[np.ndarray],
        item_popularity: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate diversity metrics."""
        results = {}
        
        all_recommended_items = []
        k = max(self.k_values)
        
        for user_idx in unique_users:
            recommended_items = self._get_recommendations(model, user_idx, k)
            all_recommended_items.extend(recommended_items)
        
        # Calculate diversity metrics
        results["intra_list_diversity"] = self.diversity_metrics.intra_list_diversity(
            all_recommended_items, item_features
        )
        
        if item_popularity is not None:
            results["novelty"] = self.diversity_metrics.novelty(
                all_recommended_items, item_popularity
            )
        
        # Coverage requires all items
        if len(test_data) > 0:
            all_items = np.unique(test_data[:, 1])
            results["coverage"] = self.diversity_metrics.coverage(
                all_recommended_items, all_items.tolist()
            )
        
        return results
    
    def _get_recommendations(
        self,
        model: torch.nn.Module,
        user_idx: int,
        k: int
    ) -> List[int]:
        """Get top-k recommendations for a user."""
        model.eval()
        
        # Get all items
        n_items = model.n_items
        
        # Create user tensor
        user_tensor = torch.tensor([user_idx], dtype=torch.long)
        
        # Get scores for all items
        with torch.no_grad():
            scores = []
            for item_idx in range(n_items):
                item_tensor = torch.tensor([item_idx], dtype=torch.long)
                score = model(user_tensor, item_tensor).item()
                scores.append((item_idx, score))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [item_idx for item_idx, _ in scores[:k]]
