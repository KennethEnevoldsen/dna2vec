from pathlib import Path
import torch

from dna2vec_sft.sft_config_schema import (
    SFTConfigSchema,
    SFTModelConfigSchema,
    SFTTrainingConfigSchema,
    MLPConfigSchema
)
from dna2vec.config_schema import (
    DatasetConfigSchemaUniformSampling,
    SchedulerConfigSchema,
)
from dna2vec_sft.dataset import FastaUniformSampler
from dna2vec_sft.sft_main import sft_main

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# SFT Configuration
CONFIG = SFTConfigSchema(
    model_config=SFTModelConfigSchema(
        pretrained_encoder_path=Path("/mnt/SSD7/yigit/dnarde/regularization_0.0_pool_1020mean_checkpoint.pt"),
        embedding_dim=1020,
        dim_feedforward=1536,
        num_heads=12,
        num_layers=6,
        dropout=0.1,
        activation="gelu",
        max_position_embeddings=1024,
        
        # MLP Head Configuration
        mlp_config=MLPConfigSchema(
            input_dim=1020,  # Should match embedding_dim
            hidden_dim=512,
            output_dim=1020,  # Final projection dimension
            dropout=0.1,
            activation="relu",
            num_layers=2
        ),
        
        # Optional: Load pre-trained encoder
        # pretrained_encoder_path=Path("path/to/pretrained/encoder.pt"),
        freeze_encoder=False,  # Set to True if you want to freeze encoder weights
    ),
    
    training_config=SFTTrainingConfigSchema(
        max_steps=50_000,  # Reduced from original for SFT
        batch_size=16,
        device=device,
        log_interval=100,
        accumulation_steps=8,  # Reduced accumulation steps
        scheduler_config=SchedulerConfigSchema(
            max_lr=1e-5,  # Lower learning rate for SFT
        ),
        regularizer=0,
        pool_type="mean",
        
        # SFT specific parameters
        warmup_steps=1000,
        patience=5,  # Early stopping patience
        min_delta=1e-4,
    ),
    
    dataset_config=DatasetConfigSchemaUniformSampling(
        fasta_file=[
            Path("/mnt/SSD7/yigit/dnarde/ch2/NC-000002.fasta")
        ],
        range_min=1500, #TODO make it 800 back
        range_max=2000,
        subsequence_range_min=150,
        subsequence_range_max=500,
        dataset=FastaUniformSampler,
        sampling_strategy="three_nonoverlapping_subsequence_uppercase", #"random_subsequence_uppercase", #TODO: Change it back
        read_regularizer=True,  # Keep this to be true, things break if its false
    ),
    
    val_dataset_config=DatasetConfigSchemaUniformSampling(
        fasta_file=[
            Path("/mnt/SSD7/yigit/dnarde/ch2/NC-000002.fasta")
        ],
        range_min=1500, #TODO make it 800 back
        range_max=2000,
        subsequence_range_min=150,
        subsequence_range_max=500,
        dataset=FastaUniformSampler,
        sampling_strategy="three_nonoverlapping_subsequence_uppercase", #"random_subsequence_uppercase", #TODO: Change it back
        read_regularizer=True,  # Keep this to be true, things break if its false
    )
)

if __name__ == "__main__":
    # Run SFT training
    sft_main(CONFIG, wandb_watch=True) 