"""
Base configurations
"""

import json
from typing import Literal, Optional, Type

import torch
from pydantic import BaseModel
from torch import nn

from dna2vec.model import SinusoidalPositionalEncoding


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


class ConfigSchema(BaseModel):
    model_config: ModelConfigSchema = ModelConfigSchema()
    training_config: TrainingConfigSchema = TrainingConfigSchema()
