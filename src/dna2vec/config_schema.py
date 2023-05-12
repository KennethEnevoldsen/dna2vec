"""
Base configurations
"""

from pathlib import Path
from typing import Literal, Type

import torch
from pydantic import BaseModel
from torch import nn
from torch.utils.data import Dataset

from dna2vec.dataset import FastaSamplerDataset
from dna2vec.model import SinusoidalPositionalEncoding

project_path = Path(__file__).parent.parent
tokenizer_path = project_path / "model" / "tokenizers" / "dna_tokenizer_10k.json"


class ModelConfigSchema(BaseModel):
    embedding_dim: int = 384
    dim_feedforward: int = 1536
    vocab_size: int = 4
    num_heads: int = 12
    num_layers: int = 6
    dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "gelu"
    pos_embedding: Type[nn.Module] = SinusoidalPositionalEncoding
    max_position_embeddings: int = 512
    tokenizer_path: Path = tokenizer_path

    class Config:
        arbitrary_types_allowed = True


class OptimizerConfigSchema(BaseModel):
    learning_rate: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    eps = 1e-8


class TrainingConfigSchema(BaseModel):
    batch_size: int = 64
    optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
    optimizer_config: OptimizerConfigSchema = OptimizerConfigSchema()

    max_steps: int = 1000
    log_interval: int = 100
    device: torch.device = torch.device("cpu")

    class Config:
        arbitrary_types_allowed = True


class DatasetConfigSchema(BaseModel):
    dataset: Type[Dataset] = FastaSamplerDataset
    fasta_file: Path = project_path / "tests" / "data" / "NC_000002.12.txt"
    range_mean: float = 200
    range_std: float = 20


class ConfigSchema(BaseModel):
    model_config: ModelConfigSchema = ModelConfigSchema()
    training_config: TrainingConfigSchema = TrainingConfigSchema()
    dataset_config: DatasetConfigSchema = DatasetConfigSchema()
