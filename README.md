# Neural Collaborative Filtering

A production-ready implementation of Neural Collaborative Filtering (NCF) for personalized recommendations using deep learning.

## Overview

Neural Collaborative Filtering is an advanced recommendation technique that uses deep learning to model complex, non-linear relationships between users and items. This implementation includes multiple NCF architectures:

- **GMF (Generalized Matrix Factorization)**: Element-wise product of user and item embeddings
- **MLP (Multi-Layer Perceptron)**: Deep neural network on concatenated embeddings  
- **NeuMF (Neural Matrix Factorization)**: Fusion of GMF and MLP components
- **BPR (Bayesian Personalized Ranking)**: Ranking-based optimization

## Features

- **Multiple Architectures**: GMF, MLP, NeuMF, and BPR models
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate@K, MRR@K
- **Diversity Metrics**: Intra-list diversity, coverage, novelty
- **Interactive Demo**: Streamlit-based web interface
- **Production Ready**: Type hints, logging, configuration management
- **Extensible**: Easy to add new models and metrics

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Neural-Collaborative-Filtering-v2.git
cd Neural-Collaborative-Filtering-v2

# Install dependencies
pip install -r requirements.txt

# Or install with pip
pip install -e .
```

### Training

```bash
# Train with default configuration
python src/train.py

# Train with custom configuration
python src/train.py model.architecture=mlp training.epochs=100

# Train with different data settings
python src/train.py data.n_users=2000 data.n_items=1000
```

### Demo

```bash
# Start the interactive demo
streamlit run src/demo.py

# Or use Gradio
python src/demo.py --interface gradio
```

## Project Structure

```
neural-collaborative-filtering/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/          # NCF model implementations
│   ├── evaluation/      # Evaluation metrics and utilities
│   ├── utils/           # Utility functions
│   ├── train.py         # Main training script
│   └── demo.py          # Demo application
├── configs/             # Configuration files
│   ├── config.yaml      # Main configuration
│   ├── model/           # Model-specific configs
│   ├── data/            # Data-specific configs
│   └── evaluation/      # Evaluation configs
├── tests/               # Unit tests
├── notebooks/           # Jupyter notebooks
├── scripts/             # Utility scripts
├── data/                # Data directory
├── checkpoints/         # Model checkpoints
└── assets/              # Static assets
```

## Configuration

The project uses Hydra for configuration management. Key configuration options:

### Model Configuration

```yaml
model:
  architecture: neumf  # gmf, mlp, neumf, bpr
  embedding_dim: 64
  hidden_dims: [128, 64, 32]
  dropout: 0.2
  use_batch_norm: true
  learning_rate: 0.001
  optimizer: adam
```

### Data Configuration

```yaml
data:
  n_users: 1000
  n_items: 500
  n_interactions: 10000
  rating_scale: [1, 5]
  sparsity: 0.95
  negative_sampling_ratio: 4
```

### Training Configuration

```yaml
training:
  batch_size: 256
  epochs: 50
  early_stopping_patience: 10
  validation_split: 0.2
  test_split: 0.1
```

## Data Format

The system expects data in the following format:

### Interactions CSV
```csv
user_id,item_id,rating,timestamp
1,101,5,1640995200
1,102,4,1640995260
2,101,3,1640995320
```

### Items CSV
```csv
item_id,title,category,features
101,Movie A,Action,"action,thriller"
102,Movie B,Comedy,"comedy,romance"
```

### Users CSV (Optional)
```csv
user_id,age,gender,location
1,25,M,US
2,30,F,EU
```

## Model Architectures

### GMF (Generalized Matrix Factorization)
- Element-wise product of user and item embeddings
- Simple but effective for capturing multiplicative interactions
- Good baseline for comparison

### MLP (Multi-Layer Perceptron)
- Deep neural network on concatenated embeddings
- Captures complex non-linear interactions
- More parameters than GMF

### NeuMF (Neural Matrix Factorization)
- Combines GMF and MLP components
- Fusion layer combines both representations
- State-of-the-art performance

### BPR (Bayesian Personalized Ranking)
- Ranking-based optimization
- Uses dot product of embeddings
- Good for implicit feedback

## Evaluation Metrics

### Ranking Metrics
- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation
- **MRR@K**: Mean Reciprocal Rank

### Diversity Metrics
- **Intra-list Diversity**: Diversity within recommendation lists
- **Coverage**: Fraction of catalog items recommended
- **Novelty**: Average popularity of recommended items

## API Reference

### Models

```python
from src.models import create_model

# Create a model
model = create_model(n_users, n_items, config)

# Forward pass
predictions = model(user_idx, item_idx)

# Get embeddings
user_emb = model.get_user_embeddings(user_idx)
item_emb = model.get_item_embeddings(item_idx)
```

### Data Loading

```python
from src.data import SyntheticDataLoader

# Load synthetic data
data_loader = SyntheticDataLoader(config)
data_loader.load_data()
data_loader.preprocess_data()

# Get train/val/test splits
train, val, test = data_loader.get_train_val_test_split()
```

### Evaluation

```python
from src.evaluation import RecommendationEvaluator

# Initialize evaluator
evaluator = RecommendationEvaluator(config)

# Evaluate model
metrics = evaluator.evaluate_model(model, test_data)
```

## Advanced Usage

### Custom Models

```python
from src.models import BaseNCFModel

class CustomNCFModel(BaseNCFModel):
    def __init__(self, n_users, n_items, config):
        super().__init__(n_users, n_items, config)
        # Add custom layers
        
    def forward(self, user_idx, item_idx):
        # Implement custom forward pass
        pass
```

### Custom Metrics

```python
from src.evaluation import RankingMetrics

class CustomRankingMetrics(RankingMetrics):
    @staticmethod
    def custom_metric(recommended, relevant, k):
        # Implement custom metric
        pass
```

### Custom Data Loaders

```python
from src.data import BaseDataLoader

class CustomDataLoader(BaseDataLoader):
    def load_data(self):
        # Implement custom data loading
        pass
```

## Performance Tips

1. **Batch Size**: Larger batch sizes generally improve training stability
2. **Learning Rate**: Start with 0.001 and adjust based on convergence
3. **Embedding Dimension**: 64-128 dimensions work well for most datasets
4. **Negative Sampling**: Use 4:1 ratio of negative to positive samples
5. **Early Stopping**: Monitor validation loss to prevent overfitting

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or embedding dimension
2. **Slow Training**: Use GPU acceleration and increase batch size
3. **Poor Performance**: Try different architectures or hyperparameters
4. **Convergence Issues**: Adjust learning rate or use learning rate scheduling

### Debug Mode

```bash
# Enable debug logging
python src/train.py log_level=DEBUG

# Run with minimal data for testing
python src/train.py data.n_users=100 data.n_items=50
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{neural_collaborative_filtering,
  title={Neural Collaborative Filtering},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Neural-Collaborative-Filtering-v2}
}
```

## Acknowledgments

- Original NCF paper: He et al., "Neural Collaborative Filtering" (WWW 2017)
- PyTorch team for the excellent deep learning framework
- Hydra team for configuration management
- Streamlit team for the demo framework
# Neural-Collaborative-Filtering-v2
