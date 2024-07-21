"""
A implementation of the contrastive siamese architecture from sentence transformers to learn DNA embeddings.
"""
import math
import numpy as np
from typing import Dict, Literal, Optional, Tuple, Type

import torch
import torch.nn as nn

from dna2vec.tokenizer  import BPTokenizer

import logging

class SinusoidalPositionalEncoding(nn.Module):
    """
    Derived from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 1024,
    ):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.squeeze(1).unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.names = ["batch", "sequence"]
        return self.pe[:, :x.size(1), :]  # type: ignore # [batch, seq_len, d_model]


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

#New

class DynamicAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_dropout=0.05):
        super(DynamicAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads  # Dimension per head
        self.scaling_factor = nn.Parameter(torch.zeros(num_heads))
        self.attention_dropout = nn.Dropout(attn_dropout)
        self.norm = nn.LayerNorm(d_model)

        # Separate linear transformations for queries, keys, and values
        self.query_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.key_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.value_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])

        # Final output linear transformation
        self.out = nn.Linear(d_model, d_model)
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Initialize linear transformations query, key, value layers and output layer
        for layer in self.query_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        for layer in self.key_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        for layer in self.value_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)  
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)
        

    def forward(self, query, key, value, mask=None):

        # Applying separate linear transformations for each head
        qs = torch.cat([layer(query).view(query.size(0), 1, query.size(1), self.d_k) for layer in self.query_layers], dim=1)
        ks = torch.cat([layer(key).view(query.size(0), 1, query.size(1), self.d_k) for layer in self.key_layers], dim=1)
        vs = torch.cat([layer(value).view(query.size(0), 1, query.size(1), self.d_k) for layer in self.value_layers], dim=1)

        # Calculate scaled dot-product attention scores
        
        scores = torch.matmul(qs, ks.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -9e15)
            
        attn = nn.functional.softmax(scores, dim=-1)
        attn = attn * self.scaling_factor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        attn = self.attention_dropout(attn) # Derived from https://arxiv.org/abs/1907.11065

        # Compute context vectors and reshape back to the original d_model dimensions
        context = torch.matmul(attn, vs).transpose(1, 2).contiguous().view(query.size(0), query.size(1), self.d_model)

        # Apply final linear transformation and dropout
        output = self.out(context)
        # Add and normalize
        return output

class FeedbackAttention(nn.Module):
    def __init__(self, d_model, num_heads, iterations=3, attn_dropout=0.05, dropout=0.1):
        super(FeedbackAttention, self).__init__()
        self.num_heads = num_heads
        self.iterations = iterations
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        # Initialize linear transformations for queries, keys, and values
        self.query_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.key_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.value_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        
        # Dropout layers for attention and output
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attn_dropout)
        
        # Output linear layer and normalization layer
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Initialize linear transformations query, key, value layers and output layer
        for layer in self.query_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        for layer in self.key_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        for layer in self.value_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)  
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x, mask=None):
        output = x
        for _ in range(self.iterations):
            # Compute query, key, and value for all heads
            qs = torch.stack([layer(output) for layer in self.query_layers], dim=1)
            ks = torch.stack([layer(x) for layer in self.key_layers], dim=1)
            vs = torch.stack([layer(x) for layer in self.value_layers], dim=1)
            
            # Calculate attention scores
            scores = torch.matmul(qs, ks.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            # Apply masking if provided
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -9e15)
            
            # Compute softmax over the last dimension to get attention probabilities
            attn = torch.nn.functional.softmax(scores, dim=-1)
            attn = self.attention_dropout(attn) # Derived from https://arxiv.org/abs/1907.11065
            
            # Compute weighted sum of values
            context = torch.matmul(attn, vs)
            context = context.transpose(1, 2).contiguous().view(output.size(0), -1, self.d_model)
            
            # Pass through the output layer and apply dropout and normalization
            output = self.out(context)
            output = self.dropout(output)
            output = self.norm(output + x)
        
        return output


class DynamicFeedbackAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_dropout=0.05, iterations=3):
        super(DynamicFeedbackAttention, self).__init__()
        self.num_heads = num_heads
        self.iterations = iterations
        self.d_model = d_model
        self.d_k = d_model // num_heads  # Dimension of each head
        self.scaling_factor = nn.Parameter(torch.zeros(num_heads))
        
        # Separate linear transformations for queries, keys, and values for each head
        self.query_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.key_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.value_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        
        self.attention_dropout = nn.Dropout(attn_dropout)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Initialize linear transformations query, key, value layers and output layer
        for layer in self.query_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        for layer in self.key_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        for layer in self.value_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)  
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x, mask=None):
        output = x
        for _ in range(self.iterations):
            # Applying separate transformations and stacking for multi-head attention
            qs = torch.stack([layer(output) for layer in self.query_layers], dim=1)
            ks = torch.stack([layer(x) for layer in self.key_layers], dim=1)
            vs = torch.stack([layer(x) for layer in self.value_layers], dim=1)

            # Calculate scaled dot-product attention scores

            scores = torch.matmul(qs, ks.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -9e15)
            
            attn = nn.functional.softmax(scores, dim=-1)
            attn = attn * self.scaling_factor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            attn = self.attention_dropout(attn) # Derived from https://arxiv.org/abs/1907.11065
            

            # Compute context vectors and reshape back to the original d_model dimensions
            context = torch.matmul(attn, vs).transpose(1, 2).contiguous().reshape(x.size(0), x.size(1), self.d_model)

            # Pass through the output layer and apply dropout and normalization
            output = self.out(context)

        return output



class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = DynamicFeedbackAttention(d_model = d_model, num_heads = nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.functional.relu if activation == "relu" else nn.functional.gelu
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)
        
    def forward(self, src, src_key_padding_mask=None):
        src2 = self.norm1(src)
        # src2 = self.self_attn(src2, src2, src2, src_key_padding_mask)
        src2 = self.self_attn(src2, src_key_padding_mask) # for feedback attn
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.activation(self.linear1(src2))
        src2 = self.dropout(src2)
        src2 = self.linear2(src2)
        src = src + self.dropout2(src2)
        return src
    
class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask)
        if self.norm:
            output = self.norm(output)
        return output

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

        self.positional_embedding = pos_embedding(
            d_model=embedding_dim,
            max_len=max_position_embeddings,
        )

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        # create encode layers
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=embedding_dim,
        #     nhead=num_heads,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout,
        #     activation=activation,
        #     batch_first=True,
        #     norm_first=True,  # following: https://arxiv.org/pdf/2002.04745.pdf
        # )
        # self.trf_encoder = nn.TransformerEncoder(
        #     encoder_layer=encoder_layer, num_layers=num_layers
        # )
        
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.trf_encoder = CustomTransformerEncoder(encoder_layer, num_layers)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
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
        # out.names = ["batch", "sequence", "embedding"]
        return out


class AveragePooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    # derived from https://github.com/princeton-nlp/SimCSE/blob/13361d0e29da1691e313a94f003e2ed1cfa97fef/simcse/models.py#LL49C1-L84C1
    """

    def __init__(self):
        super().__init__()

    def forward(self, last_hidden, attention_mask):
        # Old previous implementation
        return (last_hidden * attention_mask.unsqueeze(-1)).sum(
            1
        ) / attention_mask.sum(-1).unsqueeze(-1)

        # last_hidden.names = ["batch", "sequence", "embedding"]
        # attention_mask.names = ["batch", "sequence"]
        # # using named tensors
        # attention_mask = attention_mask.align_to(
        #     "batch", "sequence", "embedding"
        # )  # eq. to unsqueeze

        # avg_emb = (last_hidden * attention_mask).sum("sequence") / attention_mask.sum(
        #     "sequence"
        # )
        # return avg_emb


def model_from_config(cfg) -> Tuple[Encoder, nn.Module, BPTokenizer]:
    """
    Create a model from a configuration object

    Args:
        cfg: The configuration object

    Returns:
        The model, the pooling layer and the tokenizer
    """
    if cfg.model_path is None:
        logging.info("Creating new model")

        # create model
        model_kwargs = cfg.dict()
        model_kwargs.pop("model_path")
        pooler = model_kwargs.pop("pooling")
        tokenizer_path = model_kwargs.pop("tokenizer_path")
        tokenizer = BPTokenizer.load(str(tokenizer_path))

        if cfg.vocab_size is None:
            model_kwargs["vocab_size"] = tokenizer.vocab_size + 1
            cfg.vocab_size = tokenizer.vocab_size + 1

        model = Encoder(**model_kwargs)

        return model, pooler, tokenizer

    # check if model_path exists
    if not cfg.model_path.exists():
        logging.warning(f"Model path {cfg.model_path} does not exist, creating new model")
        cfg.model_path = None
        return model_from_config(cfg)

    # load model
    logging.info(f"Loading model from {cfg.model_path}, ignoring config")

    info_dict = torch.load(cfg.model_path)
    model_kwargs = info_dict["model_kwargs"]
    
    # load tokenizer
    tokenizer_path = model_kwargs.pop("tokenizer_path")
    # pooling
    pooler = model_kwargs.pop("pooling")

    tokenizer = BPTokenizer.load(str(tokenizer_path))
    model = Encoder(**model_kwargs)
    model.load_state_dict(info_dict["model_state_dict"])

    return model, pooler, tokenizer




if __name__ == "__main__":
    dna_seq = torch.randint(0, 4, (10, 30), dtype=torch.long)
    # dna_seq.names = ["batch", "sequence"]

    encoder = Encoder()
    output = encoder(dna_seq)
    print(output.shape)
