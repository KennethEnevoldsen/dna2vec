# DNA2Vec SFT with Hydra Configuration

This document explains how to use the Hydra configuration system for DNA2Vec SFT training.

## Overview

Hydra provides hierarchical configuration management, making it easy to:
- Organize configurations into reusable components
- Override parameters from the command line
- Track experiments automatically
- Switch between different experimental setups

## Configuration Structure

```
configs/
├── config.yaml                    # Main configuration
├── experiment/                    # Pre-defined experiments
│   ├── triplet_loss.yaml         # Triplet loss focused training
│   └── quick_test.yaml           # Quick testing configuration
├── model/                        # Model configurations
│   ├── sft_model.yaml            # Standard SFT model
│   └── sft_model_large.yaml      # Larger model variant
├── training/                     # Training configurations
│   ├── sft_training.yaml         # Standard training
│   └── sft_training_fast.yaml    # Fast training for experiments
├── dataset/                      # Dataset configurations
│   ├── train_dataset.yaml        # Training dataset
│   └── val_dataset.yaml          # Validation dataset
```

## Basic Usage

### 1. Run with Default Configuration

```bash
python train_hydra.py
```

### 2. Override Parameters from Command Line

```bash
# Change learning rate
python train_hydra.py training_config.optimizer_config.lr=1e-4

# Change batch size and max steps
python train_hydra.py training_config.batch_size=32 training_config.max_steps=20000

# Change model architecture
python train_hydra.py model_config.mlp_config.hidden_dim=1024 model_config.mlp_config.num_layers=3

# Disable WandB logging
python train_hydra.py wandb.mode=disabled
```

### 3. Use Pre-defined Experiments

```bash
# Run triplet loss experiment
python train_hydra.py --config-name=experiment/triplet_loss

# Run quick test
python train_hydra.py --config-name=experiment/quick_test
```

### 4. Mix and Match Components

```bash
# Use large model with fast training
python train_hydra.py model=sft_model_large training=sft_training_fast

# Use different model and training configs
python train_hydra.py model=sft_model_large training=sft_training_fast training_config.max_steps=5000
```

## Advanced Usage

### 1. Hyperparameter Sweeps

Create a sweep configuration:

```bash
# Run multiple learning rates
python train_hydra.py -m training_config.optimizer_config.lr=1e-5,5e-5,1e-4

# Grid search over model sizes and learning rates
python train_hydra.py -m model_config.mlp_config.hidden_dim=256,512,1024 training_config.optimizer_config.lr=1e-5,5e-5

# Search over loss weights
python train_hydra.py -m training_config.alpha=0.0,0.25,0.5 training_config.beta=0.5,0.75,1.0
```

### 2. Custom Experiment Naming

```bash
# Custom experiment and run names
python train_hydra.py experiment_name=my_experiment run_name=test_run_v1

# Use timestamp in name (default behavior)
python train_hydra.py run_name="my_test_${now:%Y%m%d_%H%M}"
```

### 3. Device Selection

```bash
# Specify GPU
python train_hydra.py training_config.device=cuda:1

# Auto-detect device (default)
python train_hydra.py training_config.device=auto
```

## Configuration Examples

### Example 1: Quick Debugging Session

```bash
python train_hydra.py \
  --config-name=experiment/quick_test \
  training_config.max_steps=100 \
  wandb.mode=disabled
```

### Example 2: Production Training

```bash
python train_hydra.py \
  model=sft_model_large \
  training_config.max_steps=100000 \
  training_config.batch_size=16 \
  experiment_name=production_run \
  run_name=large_model_${now:%Y%m%d}
```

### Example 3: Ablation Study

```bash
# Test different loss combinations
python train_hydra.py -m \
  training_config.alpha=0.0,0.25,0.5,1.0 \
  training_config.beta=0.0,0.25,0.5,1.0 \
  experiment_name=loss_ablation
```

## Output Organization

Hydra automatically organizes outputs:

```
outputs/
└── experiment_name/
    └── run_name/
        ├── .hydra/
        │   ├── config.yaml        # Resolved configuration
        │   ├── hydra.yaml         # Hydra configuration
        │   └── overrides.yaml     # Command line overrides
        ├── logs/                  # Training logs
        └── checkpoints/           # Model checkpoints
```

## WandB Integration

The configuration automatically integrates with WandB:

- **Project**: Set via `wandb.project`
- **Run Name**: Automatically uses `run_name` from config
- **Config Logging**: Full configuration is logged to WandB
- **Hydra Integration**: Working directory is used for WandB artifacts

```bash
# Custom WandB settings
python train_hydra.py \
  wandb.project=my_project \
  wandb.name=custom_run_name \
  wandb.mode=online
```

## Creating New Configurations

### 1. Add a New Model Configuration

Create `configs/model/my_model.yaml`:

```yaml
# @package model_config
_target_: dna2vec_sft.sft_config_schema.SFTModelConfigSchema

embedding_dim: 1020
mlp_config:
  input_dim: 1020
  hidden_dim: 2048  # Your custom size
  output_dim: 1020
  num_layers: 4     # Your custom depth
```

Use it:

```bash
python train_hydra.py model=my_model
```

### 2. Add a New Experiment

Create `configs/experiment/my_experiment.yaml`:

```yaml
# @package _global_

defaults:
  - override /model: sft_model_large
  - override /training: sft_training_fast

experiment_name: "my_experiment"
run_name: "custom_${now:%Y%m%d_%H%M}"

# Your custom overrides
training_config:
  max_steps: 25000
  alpha: 0.3
  beta: 0.7
```

Use it:

```bash
python train_hydra.py --config-name=experiment/my_experiment
```

## Tips and Best Practices

1. **Start Small**: Use `quick_test` experiment for debugging
2. **Use Sweeps**: Leverage `-m` flag for hyperparameter search
3. **Naming**: Use descriptive experiment and run names
4. **Validation**: Always validate file paths before long runs
5. **Logging**: Use WandB for experiment tracking
6. **Checkpoints**: Monitor the checkpoint directory for saved models

## Troubleshooting

### Common Issues

1. **Path Errors**: Ensure FASTA files and pretrained models exist
2. **GPU Memory**: Reduce batch size if running out of memory
3. **Configuration Errors**: Check YAML syntax and _target_ paths
4. **Import Errors**: Ensure all dependencies are installed

### Debug Mode

```bash
# Enable Hydra debug mode
python train_hydra.py --config-path=configs --config-name=config hydra.verbose=true
```

### Dry Run

```bash
# Print configuration without running
python train_hydra.py --cfg job
``` 