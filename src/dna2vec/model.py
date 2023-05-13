"""
A implementation of the contrastive siamese architecture from sentence transformers to learn DNA embeddings.
"""
import math
from typing import Dict, Literal, Optional, Type

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Derived from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
    ):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.names = ["sequence", "batch", "embedding"]
        return self.pe[: x.size(0)]  # type: ignore


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.max_len = max_len

        self.positional_embedding = nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=d_model,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.names = ["batch", "sequence"]

        # create a sequence of integers from 0 to max_seq_len of seq
        # this will be the positional embedding
        seq_length = x.shape[1]
        assert seq_length <= self.max_len, "sequence length is greater than max_seq_len"
        positions = torch.arange(seq_length)
        positions = positions.expand(x.shape[:2])

        assert positions.shape == x.shape, "positions and input tensor shape mismatch"
        x = self.positional_embedding(positions)
        # emb.names = ["batch", "sequence", "embedding"]
        return x


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
        pos_embedding: Type[nn.Module] = SinusoidalPositionalEncoding,
        max_position_embeddings: int = 512,
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

        self.positional_embedding = pos_embedding(
            d_model=embedding_dim,
            max_len=max_position_embeddings,
        )

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
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        input_ids = input_ids.rename(None)
        # input_ids.names = ["batch", "sequence"]
        # embedding does not support named tensors

        # Embed
        emb = self.emb_dropout(
            self.embedding(input_ids) + self.positional_embedding(input_ids)
        )
        # emb.names = ["batch", "sequence", "embedding"]

        # Contextualize embeddings
        attn = None
        if attention_mask is not None:
            attn = attention_mask == 0  # to boolean
        out = self.trf_encoder(emb, src_key_padding_mask=attn)
        out.names = ["batch", "sequence", "embedding"]
        return out


class AveragePooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    # derived from https://github.com/princeton-nlp/SimCSE/blob/13361d0e29da1691e313a94f003e2ed1cfa97fef/simcse/models.py#LL49C1-L84C1
    """

    def __init__(self):
        super().__init__()

    def forward(self, last_hidden, attention_mask):
        # # Old previous implementation
        # return (last_hidden * attention_mask.unsqueeze(-1)).sum(
        #     1
        # ) / attention_mask.sum(-1).unsqueeze(-1)

        last_hidden.names = ["batch", "sequence", "embedding"]
        attention_mask.names = ["batch", "sequence"]
        # using named tensors
        attention_mask = attention_mask.align_to(
            "batch", "sequence", "embedding"
        )  # eq. to unsqueeze

        avg_emb = (last_hidden * attention_mask).sum("sequence") / attention_mask.sum(
            "sequence"
        )
        return avg_emb


if __name__ == "__main__":
    dna_seq = torch.randint(0, 4, (10, 30), dtype=torch.long)
    dna_seq.names = ["batch", "sequence"]

    encoder = Encoder()
    output = encoder(dna_seq)
    print(output.shape)
    print(output.names)
