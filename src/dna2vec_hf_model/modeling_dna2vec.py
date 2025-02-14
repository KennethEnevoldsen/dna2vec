from .configuration_dna2vec import DNAEncoderConfig
from transformers import PreTrainedModel
import math
from typing import Literal, Optional
import torch
import torch.nn as nn

class Encoder(nn.Module):
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
    ):
        """
        Default values taken from miniLM v6
        https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/config.json
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.emb_dropout = nn.Dropout(p=dropout)
        
        if pos_embedding == "SinusoidalPositionalEncoding":
            position = torch.arange(max_position_embeddings).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
            )
            pe = torch.zeros(max_position_embeddings, 1, embedding_dim)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            pe = pe.squeeze(1).unsqueeze(0)
            self.register_buffer("positional_embedding", pe)
        else:
            raise ValueError(f"Positional embedding {pos_embedding} not found")

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        # create encode layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # following: https://arxiv.org/pdf/2002.04745.pdf
        )
        self.trf_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # input_ids.names = ["batch", "sequence"]
        # embedding does not support named tensors

        # Embed
        emb = self.emb_dropout(
            self.embedding(input_ids) + self.positional_embedding[:, :input_ids.size(1), :]
        )
        # emb.names = ["batch", "sequence", "embedding"]

        # Contextualize embeddings
        attn = None
        if attention_mask is not None:
            attn = attention_mask == 0  # to boolean
        out = self.trf_encoder(emb, src_key_padding_mask=attn)
        # out.names = ["batch", "sequence", "embedding"]
        return out

class DNAEncoder(PreTrainedModel):
    config_class = DNAEncoderConfig
    
    def __init__(self, config: DNAEncoderConfig):
        super().__init__(config)
        self.config = config
        self.encoder =  Encoder(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            dim_feedforward=config.dim_feedforward,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            activation=config.activation,
            max_position_embeddings=config.max_position_embeddings,
        )
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.encoder(input_ids, attention_mask)