"""Tests for Neural Collaborative Filtering models."""

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from src.data import RecommendationDataset, SyntheticDataLoader
from src.evaluation import RankingMetrics, DiversityMetrics
from src.models import GMFModel, MLPModel, NeuMFModel, NeuralCollaborativeFiltering
from src.utils import set_seed


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    config = OmegaConf.create({
        "embedding_dim": 16,
        "hidden_dims": [32, 16],
        "dropout": 0.2,
        "use_batch_norm": True,
        "architecture": "neumf"
    })
    return config


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    set_seed(42)
    
    # Create synthetic data
    data_config = OmegaConf.create({
        "n_users": 100,
        "n_items": 50,
        "n_interactions": 500,
        "rating_scale": [1, 5],
        "sparsity": 0.9
    })
    
    data_loader = SyntheticDataLoader(data_config)
    data_loader.load_data()
    data_loader.preprocess_data()
    
    return data_loader


class TestModels:
    """Test model classes."""
    
    def test_gmf_model(self, sample_config):
        """Test GMF model."""
        n_users, n_items = 100, 50
        model = GMFModel(n_users, n_items, sample_config)
        
        # Test forward pass
        user_idx = torch.tensor([0, 1])
        item_idx = torch.tensor([0, 1])
        output = model(user_idx, item_idx)
        
        assert output.shape == (2,)
        assert not torch.isnan(output).any()
    
    def test_mlp_model(self, sample_config):
        """Test MLP model."""
        n_users, n_items = 100, 50
        model = MLPModel(n_users, n_items, sample_config)
        
        # Test forward pass
        user_idx = torch.tensor([0, 1])
        item_idx = torch.tensor([0, 1])
        output = model(user_idx, item_idx)
        
        assert output.shape == (2,)
        assert not torch.isnan(output).any()
    
    def test_neumf_model(self, sample_config):
        """Test NeuMF model."""
        n_users, n_items = 100, 50
        model = NeuMFModel(n_users, n_items, sample_config)
        
        # Test forward pass
        user_idx = torch.tensor([0, 1])
        item_idx = torch.tensor([0, 1])
        output = model(user_idx, item_idx)
        
        assert output.shape == (2,)
        assert not torch.isnan(output).any()
    
    def test_ncf_model(self, sample_config):
        """Test NCF model."""
        n_users, n_items = 100, 50
        model = NeuralCollaborativeFiltering(n_users, n_items, sample_config)
        
        # Test forward pass
        user_idx = torch.tensor([0, 1])
        item_idx = torch.tensor([0, 1])
        output = model(user_idx, item_idx)
        
        assert output.shape == (2,)
        assert not torch.isnan(output).any()
    
    def test_model_embeddings(self, sample_config):
        """Test model embeddings."""
        n_users, n_items = 100, 50
        model = NeuralCollaborativeFiltering(n_users, n_items, sample_config)
        
        # Test user embeddings
        user_idx = torch.tensor([0, 1])
        user_emb = model.get_user_embeddings(user_idx)
        assert user_emb.shape == (2, sample_config.embedding_dim)
        
        # Test item embeddings
        item_idx = torch.tensor([0, 1])
        item_emb = model.get_item_embeddings(item_idx)
        assert item_emb.shape == (2, sample_config.embedding_dim)


class TestDataLoader:
    """Test data loading classes."""
    
    def test_synthetic_data_loader(self):
        """Test synthetic data loader."""
        config = OmegaConf.create({
            "n_users": 50,
            "n_items": 25,
            "n_interactions": 200,
            "rating_scale": [1, 5],
            "sparsity": 0.8
        })
        
        data_loader = SyntheticDataLoader(config)
        data_loader.load_data()
        data_loader.preprocess_data()
        
        assert data_loader.interactions is not None
        assert data_loader.user_to_idx is not None
        assert data_loader.item_to_idx is not None
        assert len(data_loader.user_to_idx) == config.n_users
        assert len(data_loader.item_to_idx) == config.n_items
    
    def test_recommendation_dataset(self, sample_data):
        """Test recommendation dataset."""
        interactions = sample_data.interactions[["user_idx", "item_idx", "rating"]]
        dataset = RecommendationDataset(interactions)
        
        assert len(dataset) > 0
        
        # Test getting item
        user_idx, item_idx, rating = dataset[0]
        assert isinstance(user_idx, torch.Tensor)
        assert isinstance(item_idx, torch.Tensor)
        assert isinstance(rating, torch.Tensor)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 3, 5, 7, 9]
        
        precision = RankingMetrics.precision_at_k(recommended, relevant, k=5)
        expected = 3 / 5  # 3 relevant items in top 5
        assert abs(precision - expected) < 1e-6
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 3, 5, 7, 9]
        
        recall = RankingMetrics.recall_at_k(recommended, relevant, k=5)
        expected = 3 / 5  # 3 out of 5 relevant items found
        assert abs(recall - expected) < 1e-6
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 3, 5, 7, 9]
        
        ndcg = RankingMetrics.ndcg_at_k(recommended, relevant, k=5)
        assert 0 <= ndcg <= 1
        assert not np.isnan(ndcg)
    
    def test_hit_rate_at_k(self):
        """Test hit rate@k calculation."""
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 3, 5, 7, 9]
        
        hit_rate = RankingMetrics.hit_rate_at_k(recommended, relevant, k=5)
        assert hit_rate == 1.0  # At least one relevant item found
        
        # Test case with no hits
        recommended_no_hit = [2, 4, 6, 8, 10]
        hit_rate_no_hit = RankingMetrics.hit_rate_at_k(recommended_no_hit, relevant, k=5)
        assert hit_rate_no_hit == 0.0
    
    def test_intra_list_diversity(self):
        """Test intra-list diversity calculation."""
        recommended = [1, 2, 3, 4, 5]
        
        # Test without features
        diversity = DiversityMetrics.intra_list_diversity(recommended)
        assert diversity == 1.0  # All items are unique
        
        # Test with duplicate items
        recommended_duplicates = [1, 1, 2, 2, 3]
        diversity_duplicates = DiversityMetrics.intra_list_diversity(recommended_duplicates)
        assert diversity_duplicates < 1.0
    
    def test_coverage(self):
        """Test coverage calculation."""
        recommended = [1, 2, 3, 4, 5]
        all_items = list(range(1, 11))  # Items 1-10
        
        coverage = DiversityMetrics.coverage(recommended, all_items)
        expected = 5 / 10  # 5 out of 10 items covered
        assert abs(coverage - expected) < 1e-6


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand1 = torch.rand(5)
        np_rand1 = np.random.rand(5)
        
        # Reset seed and generate again
        set_seed(42)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        # Should be identical
        assert torch.allclose(torch_rand1, torch_rand2)
        assert np.allclose(np_rand1, np_rand2)


if __name__ == "__main__":
    pytest.main([__file__])
