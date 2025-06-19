"""
Main SFT training script
"""

from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import wandb

from dna2vec_sft.sft_config_schema import SFTConfigSchema
from dna2vec_sft.sft_model import sft_model_from_config
from dna2vec_sft.sft_trainer import SFTTrainer
from dna2vec_sft.dataset import collate_fn_sft_triplet, collate_fn_sft
from dna2vec.utils import cfg_to_wandb_dict


def sft_main(config: SFTConfigSchema, wandb_mode: str = "online", wandb_watch: bool = False):
    """
    Main SFT training function
    
    Args:
        config: SFT configuration object
        wandb_mode: WandB logging mode
        wandb_watch: Whether to watch model with WandB
    """
    model_cfg = config.model_config
    training_cfg = config.training_config
    dataset_cfg = config.dataset_config
    
    print("=== Starting SFT Training ===")
    print(f"Device: {training_cfg.device}")
    print(f"Batch size: {training_cfg.batch_size}")
    print(f"Max steps: {training_cfg.max_steps}")
    print(f"MLP config: {model_cfg.mlp_config}")
    print(f"Collate function: {training_cfg.collate_fn}")
    
    # MODEL: Create SFT model and tokenizer
    print("Creating SFT model...")
    sft_model, tokenizer = sft_model_from_config(model_cfg)
    
    if training_cfg.collate_fn == "sft_triplet":
        _collate_fn = partial(collate_fn_sft_triplet, tokenizer=tokenizer)
    else:
        _collate_fn = partial(collate_fn_sft, tokenizer=tokenizer)
    
    print(f"SFT Model created:")
    print(f"  - Encoder embedding dim: {model_cfg.embedding_dim}")
    print(f"  - MLP input dim: {model_cfg.mlp_config.input_dim}")
    print(f"  - MLP output dim: {model_cfg.mlp_config.output_dim}")
    print(f"  - Encoder frozen: {model_cfg.freeze_encoder}")
    
    # DATASET: Dataset creation (same as original)
    print("Creating training dataset...")
    dataset_kwargs = dataset_cfg.dict()
    dataset_fn = dataset_kwargs.pop("dataset")
    train_dataset = dataset_fn(**dataset_kwargs)
    
    # VALIDATION DATASET: Create validation dataset if config provided
    val_dataloader = None
    if config.val_dataset_config is not None:
        print("Creating validation dataset...")
        val_dataset_kwargs = config.val_dataset_config.dict()
        val_dataset_fn = val_dataset_kwargs.pop("dataset")
        val_dataset = val_dataset_fn(**val_dataset_kwargs)
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=training_cfg.batch_size,
            collate_fn=_collate_fn
        )
        print(f"Validation dataloader created with batch size: {training_cfg.batch_size}")
    else:
        print("No validation dataset configuration provided - skipping validation")
    
    # TRAINING: Optimizer, scheduler, and data loader
    print("Setting up training components...")
    optimizer_cfg = training_cfg.optimizer_config
    optimizer = training_cfg.optimizer(sft_model.parameters(), **optimizer_cfg.dict())
    
    # Set total steps for scheduler if not specified
    if training_cfg.scheduler_config.total_steps is None:
        training_cfg.scheduler_config.total_steps = int(
            training_cfg.max_steps / training_cfg.accumulation_steps
        )
    
    scheduler = training_cfg.scheduler(
        optimizer, **training_cfg.scheduler_config.dict()
    )
    
    # Create training data loader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=training_cfg.batch_size, 
        collate_fn=_collate_fn
    )
    
    # Similarity function
    sim = training_cfg.similarity(temperature=training_cfg.temperature)
    
    # TRAINER: Create SFT trainer
    print("Creating SFT trainer...")
    trainer = SFTTrainer(
        sft_model=sft_model,
        similarity=sim,
        loss=training_cfg.loss,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,  # Pass validation dataloader
        scheduler=scheduler,
        device=training_cfg.device,
        config=config,
        tokenizer=tokenizer,
        regularizer=training_cfg.regularizer,
    )
    
    # WANDB: Initialize logging
    print("Initializing WandB logging...")
    wandb.init(
        project="dna2vec-sft",
        name=f"sft_mlp_{model_cfg.mlp_config.output_dim}_{training_cfg.pool_type}_w_triplet_loss_and_directional_loss",
        config=cfg_to_wandb_dict(config),
        mode=wandb_mode,
    )
    
    if wandb_watch:
        wandb.watch(sft_model, log="all", log_freq=1, log_graph=True)
    
    # TRAINING: Start training
    print("Starting training...")
    trainer.train(
        max_steps=training_cfg.max_steps,
        log_interval=training_cfg.log_interval,
    )
    
    # SAVE: Save the trained model
    save_path = training_cfg.save_path / "sft_model"
    print(f"Saving model to {save_path}")
    trainer.save_model(save_path)
    
    # After training
    final_model_path = training_cfg.save_path / "final_model_w_triplet_loss_wo_contrastive_loss.pt"
    print(f"Saving final model components to {final_model_path}")
    torch.save({
        'encoder_state_dict': trainer.sft_model.encoder.state_dict(),
        'mlp_state_dict': trainer.sft_model.mlp_head.state_dict(),
        'pooler_state_dict': trainer.sft_model.pooler.state_dict(),
    }, final_model_path)
    
    print("=== SFT Training Complete ===")


def create_sample_sft_config() -> SFTConfigSchema:
    """
    Create a sample SFT configuration for testing
    """
    from dna2vec_sft.sft_config_schema import (
        SFTConfigSchema, 
        SFTModelConfigSchema, 
        SFTTrainingConfigSchema,
        MLPConfigSchema,
        DatasetConfigSchemaUniformSampling
    )
    from src.dna2vec_sft.dataset import FastaUniformSampler
    
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    # Training dataset config
    train_dataset_config = DatasetConfigSchemaUniformSampling(
        fasta_file=[
            Path("/mnt/SSD1/shreyas/dna2vec/data/chromosome_2/NC_000002.fasta")
        ],
        range_min=1500, #TODO make it 800 back
        range_max=2000,
        subsequence_range_min=150,
        subsequence_range_max=500,
        dataset=FastaUniformSampler,
        sampling_strategy="three_nonoverlapping_subsequence_uppercase", #"random_subsequence_uppercase", #TODO: Change it back
        read_regularizer=True,
    )
    
    # Validation dataset config (can use same or different file/parameters)
    val_dataset_config = DatasetConfigSchemaUniformSampling(
        fasta_file=[
            Path("/mnt/SSD1/shreyas/dna2vec/data/chromosome_2/NC_000002.fasta")  # Same file for demo
        ],
        range_min=1500, #TODO make it 800 back
        range_max=2000,
        subsequence_range_min=150,
        subsequence_range_max=500,
        dataset=FastaUniformSampler,
        sampling_strategy="three_nonoverlapping_subsequence_uppercase", #"random_subsequence_uppercase", #TODO: Change it back
        read_regularizer=True,
    )
    
    return SFTConfigSchema(
        model_config=SFTModelConfigSchema(
            embedding_dim=1020,
            mlp_config=MLPConfigSchema(
                input_dim=1020,
                hidden_dim=512,
                output_dim=1020,
                dropout=0.1,
                activation="relu",
                num_layers=2
            ),
            # pretrained_encoder_path=Path("path/to/pretrained/encoder.pt"),  # Uncomment if you have one
            freeze_encoder=False,
        ),
        training_config=SFTTrainingConfigSchema(
            max_steps=50_000,
            batch_size=16,
            device=device,
            log_interval=100,
            accumulation_steps=8,
            scheduler_config={
                "max_lr": 5e-5,
            },
            regularizer=0,
            pool_type="mean",
            warmup_steps=1000,
            patience=5,
        ),
        dataset_config=train_dataset_config,
        val_dataset_config=val_dataset_config,  # Add validation dataset
    )


if __name__ == "__main__":
    # Create and run with sample config
    config = create_sample_sft_config()
    sft_main(config, wandb_mode="online", wandb_watch=True) 