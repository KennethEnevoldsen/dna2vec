from transformers import PretrainedConfig
from typing import Literal, Optional

class DNAEncoderConfig(PretrainedConfig):
    model_type = "dna_encoder"
    
    def __init__(
        self,
        vocab_size: int = 4,
        embedding_dim: int = 384,
        dim_feedforward: int = 1536,
        num_heads: int = 12,
        num_layers: int = 6,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu"] = "gelu",
        pos_embedding: Optional[str] = "SinusoidalPositionalEncoding",
        max_position_embeddings: int = 1024,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dim_feedforward = dim_feedforward
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.pos_embedding = pos_embedding
        self.max_position_embeddings = max_position_embeddings
        
        super().__init__(**kwargs)