# DNA2Vec SFT (Supervised Fine-Tuning) Implementation

This directory contains a supervised fine-tuning implementation that extends the original DNA2Vec model by adding a small MLP on top of the encoder while maintaining the same data loading structure and using contrastive loss.

## Architecture Overview

The SFT implementation consists of:

1. **Encoder**: The original transformer-based DNA encoder
2. **Pooler**: Pooling layer (mean or CLS pooling)  
3. **MLP Head**: A small multi-layer perceptron on top of the pooled representations
4. **Contrastive Loss**: Same contrastive learning objective as the original model

## Key Components

### Files Created:

- `sft_config_schema.py`: Extended configuration schema for SFT
- `sft_model.py`: SFT model implementation with MLP head
- `sft_trainer.py`: SFT trainer with contrastive loss and early stopping
- `sft_main.py`: Main training script for SFT
- `train_sft.py`: Simple training script (similar to original `train.py`)

### Key Features:

- **MLP Head**: Configurable multi-layer perceptron with dropout and different activations
- **Pre-trained Encoder Loading**: Optionally load a pre-trained encoder
- **Encoder Freezing**: Option to freeze encoder weights during SFT
- **Early Stopping**: Built-in early stopping with patience
- **Same Data Pipeline**: Uses the exact same data loading structure as the original
- **Contrastive Loss**: Maintains the same contrastive learning objective

## Configuration

### MLP Head Configuration:
```python
mlp_config=MLPConfigSchema(
    input_dim=1020,      # Should match encoder embedding_dim
    hidden_dim=512,      # Hidden layer dimension
    output_dim=256,      # Final projection dimension
    dropout=0.1,         # Dropout rate
    activation="relu",   # Activation function (relu, gelu, tanh)
    num_layers=2         # Number of MLP layers
)
```

### Model Configuration:
```python
model_config=SFTModelConfigSchema(
    embedding_dim=1020,
    mlp_config=mlp_config,
    pretrained_encoder_path=Path("path/to/encoder.pt"),  # Optional
    freeze_encoder=False,  # Whether to freeze encoder weights
    # ... other encoder parameters
)
```

### Training Configuration:
```python
training_config=SFTTrainingConfigSchema(
    max_steps=50_000,    # Typically fewer steps needed for SFT
    batch_size=16,       # Smaller batch size for SFT
    accumulation_steps=8, # Gradient accumulation
    warmup_steps=1000,   # Warmup steps
    patience=5,          # Early stopping patience
    min_delta=1e-4,      # Minimum improvement for early stopping
    # ... other training parameters
)
```

## Usage

### Basic Usage:

```bash
python train_sft.py
```

### Customized Usage:

```python
from train_sft import CONFIG
from sft_main import sft_main

# Modify configuration as needed
CONFIG.model_config.freeze_encoder = True  # Freeze encoder
CONFIG.model_config.mlp_config.output_dim = 128  # Smaller output dim
CONFIG.training_config.max_steps = 25_000  # Fewer training steps

# Run training
sft_main(CONFIG, wandb_watch=True)
```

### Loading Pre-trained Encoder:

```python
CONFIG.model_config.pretrained_encoder_path = Path("path/to/pretrained_encoder.pt")
CONFIG.model_config.freeze_encoder = True  # Optional: freeze pre-trained weights
```

## Key Differences from Original

1. **Model Architecture**: 
   - Original: Encoder → Pooler → Output
   - SFT: Encoder → Pooler → **MLP Head** → Output

2. **Training Configuration**:
   - Lower learning rates (5e-5 vs 1e-4)
   - Fewer training steps (50K vs 200K)
   - Early stopping with validation
   - Smaller batch sizes

3. **Additional Features**:
   - Option to freeze encoder weights
   - Pre-trained encoder loading
   - Early stopping mechanism
   - Validation monitoring

## Model Output

The SFT model outputs embeddings of dimension `mlp_config.output_dim` (default: 256) instead of the original encoder dimension (1020). This allows for:

- More compact representations
- Task-specific adaptation
- Better downstream performance

## Example Training Output

```
=== Starting SFT Training ===
Device: cuda:0
Batch size: 16
Max steps: 50000
MLP config: MLPConfigSchema(input_dim=1020, hidden_dim=512, ...)

Creating SFT model...
SFT Model created:
  - Encoder embedding dim: 1020
  - MLP input dim: 1020
  - MLP output dim: 256
  - Encoder frozen: False

Creating dataset...
Setting up training components...
Creating SFT trainer...
Initializing WandB logging...
Starting training...

Step 0, Loss: 2.3456, LR: 1.00e-06
Step 100, Loss: 1.8234, LR: 5.00e-05
...
```

## Monitoring and Logging

The SFT training logs to Weights & Biases with:
- Training loss progression
- Learning rate scheduling
- Validation loss (if validation data provided)
- Model parameters and gradients (if `wandb_watch=True`)

Project name: `dna2vec-sft`

## Notes

- The same data loading pipeline ensures compatibility with existing datasets
- Contrastive loss maintains the same learning objective as the original model
- The MLP head allows for task-specific fine-tuning while preserving learned representations
- Early stopping prevents overfitting during fine-tuning 