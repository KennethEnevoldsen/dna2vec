"""
SFT configuration schema extending the base configuration
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from dna2vec.config_schema import ConfigSchema, ModelConfigSchema, TrainingConfigSchema, DatasetConfigSchemaUniformSampling
from dna2vec_sft.dataset import FastaUniformSampler

project_path = Path(__file__).parent
tokenizer_path = (
    project_path / "src" / "model" / "tokenizers" / "dna_tokenizer_10k.json"
)


class MLPConfigSchema(BaseModel):
    """Configuration for the MLP head on top of the encoder"""
    input_dim: int = 1020  # Should match encoder embedding_dim
    hidden_dim: int = 512
    output_dim: int = 1020  # Final projection dimension
    dropout: float = 0.1
    activation: Literal["relu", "gelu", "tanh"] = "relu"
    num_layers: int = 2  # Number of layers in MLP


class SFTModelConfigSchema(ModelConfigSchema):
    """Extended model config for SFT with MLP head"""
    mlp_config: MLPConfigSchema = MLPConfigSchema()
    # Optionally load a pre-trained encoder
    pretrained_encoder_path: Optional[Path] = None
    freeze_encoder: bool = False  # Whether to freeze encoder weights during SFT


class SFTTrainingConfigSchema(TrainingConfigSchema):
    """Extended training config for SFT"""
    # Override some defaults for SFT
    batch_size: int = 32  # Slightly smaller batch size for SFT
    max_steps: int = 50_000  # Fewer steps typically needed for SFT
    
    # SFT specific parameters
    warmup_steps: int = 1000
    gradient_checkpointing: bool = False
    
    # Loss weighting (if using multiple losses)
    contrastive_loss_weight: float = 1.0
    
    # Early stopping
    patience: int = 5
    min_delta: float = 1e-4
    
    alpha: float = 0.25
    beta: float = 0.75
    gamma: float = 0.0
    collate_fn: str = "sft_triplet"


class SFTConfigSchema(ConfigSchema):
    """Main SFT configuration schema"""
    model_config: SFTModelConfigSchema = SFTModelConfigSchema()
    training_config: SFTTrainingConfigSchema = SFTTrainingConfigSchema()
    dataset_config: DatasetConfigSchemaUniformSampling = DatasetConfigSchemaUniformSampling(
        fasta_file=[Path("/mnt/SSD7/yigit/dnarde/ch2/NC-000002.fasta")],
        dataset=FastaUniformSampler
    )
    # Optional validation dataset configuration
    val_dataset_config: Optional[DatasetConfigSchemaUniformSampling] = None 