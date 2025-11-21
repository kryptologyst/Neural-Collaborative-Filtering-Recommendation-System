"""Utility scripts for Neural Collaborative Filtering."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig

from src.data import SyntheticDataLoader
from src.models import create_model
from src.utils import get_device


def generate_sample_data(output_dir: str = "data/raw") -> None:
    """Generate sample data for demonstration.
    
    Args:
        output_dir: Output directory for generated data.
    """
    logger.info("Generating sample data...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    with hydra.initialize(config_path="../configs"):
        config = hydra.compose(config_name="config")
    
    data_loader = SyntheticDataLoader(config.data)
    data_loader.load_data()
    data_loader.preprocess_data()
    
    # Save interactions
    interactions_file = output_path / "interactions.csv"
    data_loader.interactions.to_csv(interactions_file, index=False)
    logger.info(f"Saved interactions to {interactions_file}")
    
    # Save items
    if data_loader.items is not None:
        items_file = output_path / "items.csv"
        data_loader.items.to_csv(items_file, index=False)
        logger.info(f"Saved items to {items_file}")
    
    # Save users
    if data_loader.users is not None:
        users_file = output_path / "users.csv"
        data_loader.users.to_csv(users_file, index=False)
        logger.info(f"Saved users to {users_file}")
    
    logger.info("Sample data generation complete!")


def analyze_data(data_path: str) -> None:
    """Analyze dataset statistics.
    
    Args:
        data_path: Path to data directory.
    """
    logger.info(f"Analyzing data in {data_path}")
    
    data_path = Path(data_path)
    
    # Load interactions
    interactions_file = data_path / "interactions.csv"
    if not interactions_file.exists():
        logger.error(f"Interactions file not found: {interactions_file}")
        return
    
    interactions = pd.read_csv(interactions_file)
    
    # Basic statistics
    logger.info("Dataset Statistics:")
    logger.info(f"  Total interactions: {len(interactions)}")
    logger.info(f"  Unique users: {interactions['user_id'].nunique()}")
    logger.info(f"  Unique items: {interactions['item_id'].nunique()}")
    logger.info(f"  Rating range: {interactions['rating'].min()} - {interactions['rating'].max()}")
    logger.info(f"  Average rating: {interactions['rating'].mean():.2f}")
    
    # Sparsity
    total_possible = interactions['user_id'].nunique() * interactions['item_id'].nunique()
    sparsity = 1 - (len(interactions) / total_possible)
    logger.info(f"  Sparsity: {sparsity:.4f}")
    
    # User and item statistics
    user_counts = interactions['user_id'].value_counts()
    item_counts = interactions['item_id'].value_counts()
    
    logger.info(f"  User interactions - Min: {user_counts.min()}, Max: {user_counts.max()}, Mean: {user_counts.mean():.2f}")
    logger.info(f"  Item interactions - Min: {item_counts.min()}, Max: {item_counts.max()}, Mean: {item_counts.mean():.2f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Rating distribution
    axes[0, 0].hist(interactions['rating'], bins=20, alpha=0.7)
    axes[0, 0].set_title('Rating Distribution')
    axes[0, 0].set_xlabel('Rating')
    axes[0, 0].set_ylabel('Frequency')
    
    # User interaction counts
    axes[0, 1].hist(user_counts, bins=20, alpha=0.7)
    axes[0, 1].set_title('User Interaction Counts')
    axes[0, 1].set_xlabel('Number of Interactions')
    axes[0, 1].set_ylabel('Number of Users')
    
    # Item interaction counts
    axes[1, 0].hist(item_counts, bins=20, alpha=0.7)
    axes[1, 0].set_title('Item Interaction Counts')
    axes[1, 0].set_xlabel('Number of Interactions')
    axes[1, 0].set_ylabel('Number of Items')
    
    # Temporal distribution (if timestamp available)
    if 'timestamp' in interactions.columns:
        interactions['date'] = pd.to_datetime(interactions['timestamp'], unit='s')
        daily_counts = interactions.groupby(interactions['date'].dt.date).size()
        axes[1, 1].plot(daily_counts.index, daily_counts.values)
        axes[1, 1].set_title('Daily Interaction Counts')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Interactions')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'No timestamp data', ha='center', va='center')
        axes[1, 1].set_title('Temporal Distribution')
    
    plt.tight_layout()
    plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
    logger.info("Saved data analysis plot to data_analysis.png")


def compare_models(config_path: str = "configs") -> None:
    """Compare different model architectures.
    
    Args:
        config_path: Path to configuration directory.
    """
    logger.info("Comparing model architectures...")
    
    # Model configurations to compare
    model_configs = {
        "GMF": {"architecture": "gmf"},
        "MLP": {"architecture": "mlp"},
        "NeuMF": {"architecture": "neumf"},
        "BPR": {"architecture": "bpr"}
    }
    
    results = {}
    
    for model_name, model_config in model_configs.items():
        logger.info(f"Training {model_name} model...")
        
        # Create configuration
        with hydra.initialize(config_path=config_path):
            config = hydra.compose(config_name="config", overrides=[f"model={model_name.lower()}"])
        
        # Setup data
        data_loader = SyntheticDataLoader(config.data)
        data_loader.load_data()
        data_loader.preprocess_data()
        
        # Create model
        n_users = len(data_loader.user_to_idx)
        n_items = len(data_loader.item_to_idx)
        model = create_model(n_users, n_items, config.model)
        
        # Get train/test split
        train_data, val_data, test_data = data_loader.get_train_val_test_split()
        
        # Simple training (for demo purposes)
        device = get_device("cpu")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Convert to tensors
        train_tensors = torch.tensor(train_data[["user_idx", "item_idx", "rating"]].values)
        
        # Train for a few epochs
        model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            user_idx = train_tensors[:, 0].long()
            item_idx = train_tensors[:, 1].long()
            rating = train_tensors[:, 2].float()
            
            predicted = model(user_idx, item_idx)
            loss = criterion(predicted, rating)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_tensors = torch.tensor(test_data[["user_idx", "item_idx", "rating"]].values)
            user_idx = test_tensors[:, 0].long()
            item_idx = test_tensors[:, 1].long()
            rating = test_tensors[:, 2].float()
            
            predicted = model(user_idx, item_idx)
            mse = criterion(predicted, rating).item()
            
            # Calculate MAE
            mae = torch.mean(torch.abs(predicted - rating)).item()
            
            results[model_name] = {"MSE": mse, "MAE": mae}
    
    # Display results
    logger.info("Model Comparison Results:")
    results_df = pd.DataFrame(results).T
    print(results_df)
    
    # Save results
    results_df.to_csv("model_comparison.csv")
    logger.info("Saved results to model_comparison.csv")


def export_model(model_path: str, output_path: str) -> None:
    """Export model for deployment.
    
    Args:
        model_path: Path to model checkpoint.
        output_path: Output path for exported model.
    """
    logger.info(f"Exporting model from {model_path} to {output_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Extract model state
    model_state = checkpoint["model_state_dict"]
    config = checkpoint["config"]
    
    # Create export data
    export_data = {
        "model_state_dict": model_state,
        "config": config,
        "model_info": {
            "architecture": config.model.get("architecture", "neumf"),
            "embedding_dim": config.model.get("embedding_dim", 64),
            "hidden_dims": config.model.get("hidden_dims", [128, 64, 32])
        }
    }
    
    # Save export
    torch.save(export_data, output_path)
    logger.info(f"Model exported to {output_path}")


def main():
    """Main script entry point."""
    parser = argparse.ArgumentParser(description="Neural Collaborative Filtering Utilities")
    parser.add_argument("command", choices=["generate_data", "analyze_data", "compare_models", "export_model"],
                       help="Command to execute")
    parser.add_argument("--data_path", default="data/raw", help="Path to data directory")
    parser.add_argument("--model_path", help="Path to model checkpoint")
    parser.add_argument("--output_path", help="Output path for export")
    
    args = parser.parse_args()
    
    if args.command == "generate_data":
        generate_sample_data(args.data_path)
    elif args.command == "analyze_data":
        analyze_data(args.data_path)
    elif args.command == "compare_models":
        compare_models()
    elif args.command == "export_model":
        if not args.model_path or not args.output_path:
            logger.error("Model path and output path required for export")
            return
        export_model(args.model_path, args.output_path)


if __name__ == "__main__":
    main()
