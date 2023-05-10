"""
A implementation of the contrastive siamese architecture from sentence transformers to learn DNA embeddings.
"""
from typing import Literal, Optional

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 384,
        dim_feedforward: int = 1536,
        vocab_size: int = 4,
        max_position_embeddings: int = 512,
        num_heads: int = 12,
        num_layers: int = 6,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu"] = "gelu",
        device: Optional[torch.device] = None,
    ):
        """
        Default values taken from miniLM v6
        https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/config.json
        """
        super(Encoder, self).__init__()
        self.max_seq_len = max_position_embeddings
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_layers = num_layers

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        # learned positional embeddings, we should probably try with something else
        self.positional_embedding = nn.Embedding(
            num_embeddings=max_position_embeddings,
            embedding_dim=embedding_dim,
        )

        # create encode layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.trf_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers, mask_check=False
        )

    def get_positional_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        get positional embeddings for the input tensor

        Args:
            x (torch.Tensor): input tensor of shape (batch, seq)

        Returns:
            torch.Tensor: positional embeddings of shape (batch, seq, embedding_dim)
        """
        # x.names = ["batch", "sequence"]

        # create a sequence of integers from 0 to max_seq_len of seq
        # this will be the positional embedding
        seq_length = x.shape[1]
        assert (
            seq_length <= self.max_seq_len
        ), "sequence length is greater than max_seq_len"
        positions = torch.arange(seq_length, device=self.device)
        positions = positions.expand(x.shape[:2])

        assert positions.shape == x.shape, "positions and input tensor shape mismatch"
        emb = self.positional_embedding(positions)
        # emb.names = ["batch", "sequence", "embedding"]
        return emb

    def forward(self, x):
        # x.names = ["batch", "sequence"]
        # embedding does not support named tensors
        x = x.rename(None)

        # Embed
        x = self.embedding(x) + self.get_positional_embedding(x)

        # x.names = ["batch", "sequence", "embedding"]

        # Contextualize embeddings
        x = self.trf_encoder(x)
        x.names = ["batch", "sequence", "embedding"]
        return x


if __name__ == "__main__":
    dna_seq = torch.randint(0, 4, (10, 30), dtype=torch.long)
    dna_seq.names = ["batch", "sequence"]

    encoder = Encoder()
    output = encoder(dna_seq)
    print(output.shape)
    print(output.names)
