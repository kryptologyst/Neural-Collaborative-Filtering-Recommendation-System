"""Main training script for Neural Collaborative Filtering."""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import RecommendationDataset, SyntheticDataLoader
from src.evaluation import RecommendationEvaluator
from src.models import create_model
from src.utils import EarlyStopping, get_device, set_seed


class NCFTrainer:
    """Trainer class for Neural Collaborative Filtering models."""
    
    def __init__(self, config: DictConfig):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.device = get_device(config.device)
        
        # Set random seed
        set_seed(config.seed)
        
        # Initialize components
        self.data_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.evaluator = None
        self.early_stopping = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
    
    def setup_data(self) -> None:
        """Setup data loading."""
        logger.info("Setting up data...")
        
        # Initialize data loader
        self.data_loader = SyntheticDataLoader(self.config.data)
        self.data_loader.load_data()
        self.data_loader.preprocess_data()
        
        # Get train/val/test splits
        self.train_data, self.val_data, self.test_data = (
            self.data_loader.get_train_val_test_split(
                temporal=self.config.data.temporal_split,
                test_ratio=self.config.training.test_split,
                val_ratio=self.config.training.validation_split
            )
        )
        
        # Add negative samples to training data
        self.train_data = self.data_loader.add_negative_samples(
            self.train_data, self.config.data.negative_sampling_ratio
        )
        
        logger.info(f"Data setup complete:")
        logger.info(f"  Train: {len(self.train_data)} interactions")
        logger.info(f"  Validation: {len(self.val_data)} interactions")
        logger.info(f"  Test: {len(self.test_data)} interactions")
    
    def setup_model(self) -> None:
        """Setup model, optimizer, and loss function."""
        logger.info("Setting up model...")
        
        # Get number of users and items
        n_users = len(self.data_loader.user_to_idx)
        n_items = len(self.data_loader.item_to_idx)
        
        # Create model
        self.model = create_model(n_users, n_items, self.config.model)
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        optimizer_name = self.config.model.get("optimizer", "adam").lower()
        learning_rate = self.config.model.get("learning_rate", 0.001)
        weight_decay = self.config.model.get("weight_decay", 1e-5)
        
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        elif optimizer_name == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Setup scheduler
        scheduler_name = self.config.model.get("scheduler", "cosine").lower()
        if scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.epochs
            )
        elif scheduler_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.5
            )
        elif scheduler_name == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=5, factor=0.5
            )
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        
        # Setup evaluator
        self.evaluator = RecommendationEvaluator(self.config.evaluation)
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.training.early_stopping_patience,
            mode="min"
        )
        
        logger.info(f"Model setup complete:")
        logger.info(f"  Model: {self.model.__class__.__name__}")
        logger.info(f"  Optimizer: {optimizer_name}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (user_idx, item_idx, rating) in enumerate(pbar):
            # Move to device
            user_idx = user_idx.to(self.device)
            item_idx = item_idx.to(self.device)
            rating = rating.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_rating = self.model(user_idx, item_idx)
            loss = self.criterion(predicted_rating, rating)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Tuple of (validation loss, validation metrics).
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for user_idx, item_idx, rating in val_loader:
                # Move to device
                user_idx = user_idx.to(self.device)
                item_idx = item_idx.to(self.device)
                rating = rating.to(self.device)
                
                # Forward pass
                predicted_rating = self.model(user_idx, item_idx)
                loss = self.criterion(predicted_rating, rating)
                
                total_loss += loss.item()
                num_batches += 1
        
        val_loss = total_loss / num_batches
        
        # Calculate validation metrics
        val_data_array = self.val_data[["user_idx", "item_idx", "rating"]].values
        val_metrics = self.evaluator.evaluate_model(
            self.model, val_data_array
        )
        
        return val_loss, val_metrics
    
    def train(self) -> None:
        """Train the model."""
        logger.info("Starting training...")
        
        # Create data loaders
        train_dataset = RecommendationDataset(
            self.train_data,
            include_negative=True,
            negative_ratio=self.config.data.negative_sampling_ratio
        )
        val_dataset = RecommendationDataset(
            self.val_data,
            include_negative=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Training loop
        start_time = time.time()
        best_val_loss = float("inf")
        
        for epoch in range(self.config.training.epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Log progress
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            logger.info(f"Epoch {epoch+1}/{self.config.training.epochs}:")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Val NDCG@10: {val_metrics.get('ndcg@10', 0):.4f}")
            logger.info(f"  Learning Rate: {current_lr:.6f}")
            logger.info(f"  Time: {epoch_time:.2f}s")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on test data."""
        logger.info("Evaluating model...")
        
        test_data_array = self.test_data[["user_idx", "item_idx", "rating"]].values
        test_metrics = self.evaluator.evaluate_model(
            self.model, test_data_array
        )
        
        logger.info("Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return test_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_metrics": self.val_metrics,
            "config": self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = Path("checkpoints") / f"checkpoint_epoch_{epoch}.pth"
        checkpoint_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path("checkpoints") / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.val_metrics = checkpoint["val_metrics"]
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function."""
    logger.info("Starting Neural Collaborative Filtering training")
    logger.info(f"Configuration: {config}")
    
    # Initialize trainer
    trainer = NCFTrainer(config)
    
    # Setup data and model
    trainer.setup_data()
    trainer.setup_model()
    
    # Train model
    trainer.train()
    
    # Evaluate model
    test_metrics = trainer.evaluate()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
