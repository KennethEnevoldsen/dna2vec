"""
DNA2Vec SFT (Supervised Fine-Tuning) Module

This module provides supervised fine-tuning capabilities for DNA2Vec models
with MLP heads and contrastive learning.
"""

# Configuration schemas
from .sft_config_schema import (
    MLPConfigSchema,
    SFTModelConfigSchema,
    SFTTrainingConfigSchema,
    SFTConfigSchema,
)

# Model components
from .sft_model import (
    MLPHead,
    SFTModel,
    sft_model_from_config,
    load_pretrained_encoder,
)

# Training
from .sft_trainer import SFTTrainer

# Main training function
from .sft_main import sft_main, create_sample_sft_config

# Training script config
from .train_sft import CONFIG as DEFAULT_SFT_CONFIG

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "MLPConfigSchema",
    "SFTModelConfigSchema", 
    "SFTTrainingConfigSchema",
    "SFTConfigSchema",
    "DEFAULT_SFT_CONFIG",
    
    # Model
    "MLPHead",
    "SFTModel",
    "sft_model_from_config",
    "load_pretrained_encoder",
    
    # Training
    "SFTTrainer",
    "sft_main",
    "create_sample_sft_config",
]

# Convenience functions for quick setup
def quick_sft_setup(
    embedding_dim: int = 1020,
    mlp_hidden_dim: int = 512,
    mlp_output_dim: int = 256,
    max_steps: int = 50_000,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    device: str = "cuda:0",
    fasta_file: str = "/mnt/SSD1/shreyas/dna2vec/data/chromosome_2/NC_000002.fasta",
    freeze_encoder: bool = False,
    pretrained_encoder_path: str = None,
) -> SFTConfigSchema:
    """
    Quick setup function for SFT configuration
    
    Args:
        embedding_dim: Encoder embedding dimension
        mlp_hidden_dim: MLP hidden layer dimension
        mlp_output_dim: MLP output dimension
        max_steps: Maximum training steps
        batch_size: Training batch size
        learning_rate: Learning rate
        device: Device to use for training
        fasta_file: Path to FASTA file
        freeze_encoder: Whether to freeze encoder weights
        pretrained_encoder_path: Path to pretrained encoder (optional)
    
    Returns:
        SFTConfigSchema: Configured SFT configuration
    """
    import torch
    from pathlib import Path
    from dna2vec_sft.dataset import FastaUniformSampler
    from src.dna2vec.config_schema import SchedulerConfigSchema, DatasetConfigSchemaUniformSampling
    
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    
    mlp_config = MLPConfigSchema(
        input_dim=embedding_dim,
        hidden_dim=mlp_hidden_dim,
        output_dim=mlp_output_dim,
        dropout=0.1,
        activation="relu",
        num_layers=2
    )
    
    model_config = SFTModelConfigSchema(
        embedding_dim=embedding_dim,
        mlp_config=mlp_config,
        freeze_encoder=freeze_encoder,
        pretrained_encoder_path=Path(pretrained_encoder_path) if pretrained_encoder_path else None,
    )
    
    training_config = SFTTrainingConfigSchema(
        max_steps=max_steps,
        batch_size=batch_size,
        device=device_obj,
        scheduler_config=SchedulerConfigSchema(max_lr=learning_rate),
        log_interval=100,
        accumulation_steps=8,
        pool_type="mean",
        warmup_steps=1000,
        patience=5,
    )
    
    dataset_config = DatasetConfigSchemaUniformSampling(
        fasta_file=[Path(fasta_file)],
        range_min=800,
        range_max=2000,
        subsequence_range_min=150,
        subsequence_range_max=500,
        dataset=FastaUniformSampler,
        sampling_strategy="random_subsequence_uppercase",
        read_regularizer=True,
    )
    
    return SFTConfigSchema(
        model_config=model_config,
        training_config=training_config,
        dataset_config=dataset_config,
    )


def train_sft_model(config: SFTConfigSchema = None, **kwargs) -> None:
    """
    Convenience function to train an SFT model
    
    Args:
        config: SFT configuration (if None, uses quick_sft_setup with kwargs)
        **kwargs: Arguments for quick_sft_setup if config is None
    """
    if config is None:
        config = quick_sft_setup(**kwargs)
    
    sft_main(config, wandb_watch=True)
