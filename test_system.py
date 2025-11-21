#!/usr/bin/env python3
"""Test script to verify the Neural Collaborative Filtering system works."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from data import SyntheticDataLoader, RecommendationDataset
        print("✅ Data modules imported successfully")
    except ImportError as e:
        print(f"❌ Data import failed: {e}")
        return False
    
    try:
        from models import GMFModel, MLPModel, NeuMFModel, NeuralCollaborativeFiltering
        print("✅ Model modules imported successfully")
    except ImportError as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    try:
        from evaluation import RankingMetrics, DiversityMetrics, RecommendationEvaluator
        print("✅ Evaluation modules imported successfully")
    except ImportError as e:
        print(f"❌ Evaluation import failed: {e}")
        return False
    
    try:
        from utils import set_seed, get_device, EarlyStopping
        print("✅ Utility modules imported successfully")
    except ImportError as e:
        print(f"❌ Utility import failed: {e}")
        return False
    
    return True


def test_data_generation():
    """Test data generation."""
    print("\nTesting data generation...")
    
    try:
        from data import SyntheticDataLoader
        from omegaconf import OmegaConf
        
        # Create minimal config
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
        assert len(data_loader.interactions) > 0
        assert data_loader.user_to_idx is not None
        assert data_loader.item_to_idx is not None
        
        print("✅ Data generation successful")
        return True, data_loader
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False, None


def test_model_creation(data_loader):
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        from models import GMFModel, MLPModel, NeuMFModel
        from omegaconf import OmegaConf
        import torch
        
        n_users = len(data_loader.user_to_idx)
        n_items = len(data_loader.item_to_idx)
        
        # Create model config
        config = OmegaConf.create({
            "embedding_dim": 16,
            "hidden_dims": [32, 16],
            "dropout": 0.2,
            "use_batch_norm": True
        })
        
        # Test different models
        models = {
            "GMF": GMFModel(n_users, n_items, config),
            "MLP": MLPModel(n_users, n_items, config),
            "NeuMF": NeuMFModel(n_users, n_items, config)
        }
        
        # Test forward pass
        user_idx = torch.tensor([0, 1])
        item_idx = torch.tensor([0, 1])
        
        for name, model in models.items():
            output = model(user_idx, item_idx)
            assert output.shape == (2,)
            assert not torch.isnan(output).any()
            print(f"✅ {name} model works")
        
        return True, models
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False, None


def test_training(models):
    """Test basic training."""
    print("\nTesting basic training...")
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Use NeuMF model for training test
        model = models["NeuMF"]
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Create dummy data
        user_idx = torch.tensor([0, 1, 2])
        item_idx = torch.tensor([0, 1, 2])
        rating = torch.tensor([3.0, 4.0, 2.0])
        
        # Training step
        optimizer.zero_grad()
        predicted = model(user_idx, item_idx)
        loss = criterion(predicted, rating)
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0
        print("✅ Basic training successful")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


def test_evaluation():
    """Test evaluation metrics."""
    print("\nTesting evaluation metrics...")
    
    try:
        from evaluation import RankingMetrics, DiversityMetrics
        
        # Test ranking metrics
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 3, 5, 7, 9]
        
        precision = RankingMetrics.precision_at_k(recommended, relevant, k=5)
        recall = RankingMetrics.recall_at_k(recommended, relevant, k=5)
        ndcg = RankingMetrics.ndcg_at_k(recommended, relevant, k=5)
        
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= ndcg <= 1
        
        print("✅ Ranking metrics work")
        
        # Test diversity metrics
        diversity = DiversityMetrics.intra_list_diversity(recommended)
        coverage = DiversityMetrics.coverage(recommended, list(range(1, 11)))
        
        assert 0 <= diversity <= 1
        assert 0 <= coverage <= 1
        
        print("✅ Diversity metrics work")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Neural Collaborative Filtering Test Suite")
    print("=" * 50)
    
    # Run tests sequentially
    results = []
    
    # Test imports
    print(f"\n🔍 Running Imports test...")
    success = test_imports()
    results.append(("Imports", success))
    
    if not success:
        print("❌ Import test failed, skipping other tests")
        return 1
    
    # Test data generation
    print(f"\n🔍 Running Data Generation test...")
    success, data_loader = test_data_generation()
    results.append(("Data Generation", success))
    
    if not success or data_loader is None:
        print("❌ Data generation test failed, skipping model tests")
        return 1
    
    # Test model creation
    print(f"\n🔍 Running Model Creation test...")
    success, models = test_model_creation(data_loader)
    results.append(("Model Creation", success))
    
    if not success or models is None:
        print("❌ Model creation test failed, skipping training test")
        return 1
    
    # Test training
    print(f"\n🔍 Running Training test...")
    success = test_training(models)
    results.append(("Training", success))
    
    # Test evaluation
    print(f"\n🔍 Running Evaluation test...")
    success = test_evaluation()
    results.append(("Evaluation", success))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
