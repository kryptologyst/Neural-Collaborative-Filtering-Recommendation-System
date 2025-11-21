"""Streamlit demo application for Neural Collaborative Filtering."""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import streamlit as st
import torch
from loguru import logger
from omegaconf import DictConfig

from src.data import SyntheticDataLoader
from src.models import create_model
from src.utils import get_device, set_seed


class NCFDemo:
    """Demo application for Neural Collaborative Filtering."""
    
    def __init__(self, config: DictConfig):
        """Initialize demo.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.device = get_device("cpu")  # Use CPU for demo
        
        # Initialize components
        self.data_loader = None
        self.model = None
        self.user_to_idx = None
        self.idx_to_user = None
        self.item_to_idx = None
        self.idx_to_item = None
        
        # Demo data
        self.demo_users = []
        self.demo_items = []
        self.user_interactions = {}
    
    def setup_data(self) -> None:
        """Setup demo data."""
        logger.info("Setting up demo data...")
        
        # Initialize data loader
        self.data_loader = SyntheticDataLoader(self.config.data)
        self.data_loader.load_data()
        self.data_loader.preprocess_data()
        
        # Get mappings
        self.user_to_idx = self.data_loader.user_to_idx
        self.idx_to_user = self.data_loader.idx_to_user
        self.item_to_idx = self.data_loader.item_to_idx
        self.idx_to_item = self.data_loader.item_to_idx
        
        # Get demo users and items
        self.demo_users = list(self.user_to_idx.keys())[:20]  # First 20 users
        self.demo_items = list(self.item_to_idx.keys())[:50]  # First 50 items
        
        # Create user interaction history
        interactions = self.data_loader.interactions
        for _, row in interactions.iterrows():
            user_id = row["user_id"]
            item_id = row["item_id"]
            rating = row["rating"]
            
            if user_id not in self.user_interactions:
                self.user_interactions[user_id] = []
            self.user_interactions[user_id].append((item_id, rating))
        
        logger.info(f"Demo data setup complete: {len(self.demo_users)} users, {len(self.demo_items)} items")
    
    def load_model(self, model_path: str) -> None:
        """Load trained model.
        
        Args:
            model_path: Path to model checkpoint.
        """
        if not Path(model_path).exists():
            logger.warning(f"Model path {model_path} does not exist, using random model")
            self._create_random_model()
            return
        
        logger.info(f"Loading model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model configuration
        model_config = checkpoint["config"].model
        
        # Create model
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        self.model = create_model(n_users, n_items, model_config)
        self.model = self.model.to(self.device)
        
        # Load state dict
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def _create_random_model(self) -> None:
        """Create a random model for demo purposes."""
        logger.info("Creating random model for demo")
        
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        
        # Create a simple model
        from src.models import MLPModel
        self.model = MLPModel(n_users, n_items, self.config.model)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def get_recommendations(self, user_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """Get recommendations for a user.
        
        Args:
            user_id: User ID.
            k: Number of recommendations.
            
        Returns:
            List of (item_id, score) tuples.
        """
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        user_tensor = torch.tensor([user_idx], dtype=torch.long)
        
        # Get scores for all items
        scores = []
        with torch.no_grad():
            for item_id in self.demo_items:
                item_idx = self.item_to_idx[item_id]
                item_tensor = torch.tensor([item_idx], dtype=torch.long)
                score = self.model(user_tensor, item_tensor).item()
                scores.append((item_id, score))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def get_similar_items(self, item_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """Get similar items to a given item.
        
        Args:
            item_id: Item ID.
            k: Number of similar items.
            
        Returns:
            List of (item_id, similarity_score) tuples.
        """
        if item_id not in self.item_to_idx:
            return []
        
        item_idx = self.item_to_idx[item_id]
        item_embedding = self.model.get_item_embeddings(torch.tensor([item_idx]))
        
        # Calculate similarities with all items
        similarities = []
        with torch.no_grad():
            for other_item_id in self.demo_items:
                if other_item_id == item_id:
                    continue
                
                other_item_idx = self.item_to_idx[other_item_id]
                other_embedding = self.model.get_item_embeddings(torch.tensor([other_item_idx]))
                
                # Cosine similarity
                similarity = torch.cosine_similarity(item_embedding, other_embedding).item()
                similarities.append((other_item_id, similarity))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def get_user_history(self, user_id: int) -> List[Tuple[int, int]]:
        """Get user's interaction history.
        
        Args:
            user_id: User ID.
            
        Returns:
            List of (item_id, rating) tuples.
        """
        return self.user_interactions.get(user_id, [])


def main():
    """Main demo function."""
    st.set_page_config(
        page_title="Neural Collaborative Filtering Demo",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Neural Collaborative Filtering Demo")
    st.markdown("Explore personalized recommendations using deep learning!")
    
    # Initialize demo
    with hydra.initialize(config_path="../configs"):
        config = hydra.compose(config_name="config")
    
    demo = NCFDemo(config)
    demo.setup_data()
    
    # Try to load trained model, fallback to random model
    model_path = "checkpoints/best_model.pth"
    demo.load_model(model_path)
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # User selection
    selected_user = st.sidebar.selectbox(
        "Select User",
        options=demo.demo_users,
        format_func=lambda x: f"User {x}"
    )
    
    # Number of recommendations
    n_recommendations = st.sidebar.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 User Profile")
        
        # User history
        user_history = demo.get_user_history(selected_user)
        if user_history:
            st.subheader("Interaction History")
            history_df = pd.DataFrame(user_history, columns=["Item ID", "Rating"])
            history_df["Item Name"] = history_df["Item ID"].apply(lambda x: f"Item {x}")
            st.dataframe(history_df[["Item Name", "Rating"]], use_container_width=True)
        else:
            st.info("No interaction history available for this user.")
        
        # Recommendations
        st.header("🎯 Recommendations")
        recommendations = demo.get_recommendations(selected_user, n_recommendations)
        
        if recommendations:
            rec_df = pd.DataFrame(recommendations, columns=["Item ID", "Score"])
            rec_df["Item Name"] = rec_df["Item ID"].apply(lambda x: f"Item {x}")
            rec_df["Rank"] = range(1, len(rec_df) + 1)
            rec_df = rec_df[["Rank", "Item Name", "Score"]]
            
            st.dataframe(rec_df, use_container_width=True)
            
            # Visualization
            st.subheader("📈 Recommendation Scores")
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(recommendations)), [score for _, score in recommendations])
            ax.set_xlabel("Rank")
            ax.set_ylabel("Score")
            ax.set_title("Recommendation Scores")
            ax.set_xticks(range(len(recommendations)))
            ax.set_xticklabels([f"Item {item_id}" for item_id, _ in recommendations], rotation=45)
            
            st.pyplot(fig)
        else:
            st.warning("No recommendations available for this user.")
    
    with col2:
        st.header("🔍 Item Similarity")
        
        # Item selection
        selected_item = st.selectbox(
            "Select Item",
            options=demo.demo_items,
            format_func=lambda x: f"Item {x}"
        )
        
        # Similar items
        similar_items = demo.get_similar_items(selected_item, 5)
        
        if similar_items:
            st.subheader(f"Items Similar to Item {selected_item}")
            for i, (item_id, similarity) in enumerate(similar_items, 1):
                st.write(f"{i}. Item {item_id} (similarity: {similarity:.3f})")
        else:
            st.info("No similar items found.")
        
        # Model info
        st.header("ℹ️ Model Information")
        st.write(f"**Architecture:** {demo.model.__class__.__name__}")
        st.write(f"**Users:** {len(demo.user_to_idx)}")
        st.write(f"**Items:** {len(demo.item_to_idx)}")
        st.write(f"**Parameters:** {sum(p.numel() for p in demo.model.parameters()):,}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with PyTorch and Streamlit | "
        "Neural Collaborative Filtering for Personalized Recommendations"
    )


if __name__ == "__main__":
    main()
