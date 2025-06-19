"""
SFT Model implementation with MLP head on top of the encoder
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from pathlib import Path

from dna2vec.model import Encoder
from dna2vec.tokenizer import BPTokenizer
from dna2vec_sft.sft_config_schema import MLPConfigSchema, SFTModelConfigSchema
from transformers import AutoModel, AutoTokenizer

class HFModel:
    def __init__(self, tokenizer, model, pooling, device):

        self.tokenizer = tokenizer
        self.model = model

        self.pooling = pooling
        self.pooling = self.pooling.to(device)

        self.model = self.model.to(device)
        self.device = device

class MLPHead(nn.Module):
    """
    Multi-layer perceptron head for SFT on top of the encoder
    """
    
    def __init__(self, config: MLPConfigSchema):
        super().__init__()
        self.config = config
        
        layers = []
        input_dim = config.input_dim
        
        # Create multiple layers
        for i in range(config.num_layers):
            if i == config.num_layers - 1:  # Last layer
                layers.append(nn.Linear(input_dim, config.output_dim))
            else:
                layers.append(nn.Linear(input_dim, config.hidden_dim))
                
                # Add activation
                if config.activation == "relu":
                    layers.append(nn.ReLU())
                elif config.activation == "gelu":
                    layers.append(nn.GELU())
                elif config.activation == "tanh":
                    layers.append(nn.Tanh())
                    
                # Add dropout
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))
                    
                input_dim = config.hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP head
        
        Args:
            x: Input tensor from encoder pooling [batch_size, embedding_dim]
            
        Returns:
            Output tensor [batch_size, output_dim]
        """
        return self.mlp(x)


class SFTModel(nn.Module):
    """
    SFT Model that combines encoder + pooler + MLP head
    """
    
    def __init__(
        self, 
        encoder: Encoder, 
        pooler: nn.Module, 
        mlp_head: MLPHead,
        freeze_encoder: bool = False
    ):
        super().__init__()
        self.encoder = encoder
        self.pooler = pooler
        self.mlp_head = mlp_head
        
        # Optionally freeze encoder weights
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        pool_type: str = "mean"
    ) -> torch.Tensor:
        """
        Forward pass through the entire SFT model
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            pool_type: Pooling type ("mean" or "cls")
            
        Returns:
            Final embeddings [batch_size, output_dim]
        """
        # Get encoder outputs
        last_hidden_state = self.encoder(input_ids, attention_mask)
        
        # Apply pooling
        if pool_type == "cls":
            pooled_output = last_hidden_state[:, 0, :]  # Use CLS token
            pooled_output = self.pooler(pooled_output)
        elif pool_type == "mean":
            pooled_output = self.pooler(last_hidden_state, attention_mask)
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}")
        
        pooled_output = torch.nn.functional.normalize(pooled_output, p=2, dim=1) # shape: (batch_size, embedding_dim)
        # Apply MLP head
        final_output = self.mlp_head(pooled_output)
        
        return final_output
    
    def get_encoder_output(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get just the encoder output (useful for analysis)
        """
        return self.encoder(input_ids, attention_mask)
    
    def get_pooled_output(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        pool_type: str = "mean"
    ) -> torch.Tensor:
        """
        Get pooled output before MLP head (useful for analysis)
        """
        last_hidden_state = self.encoder(input_ids, attention_mask)
        
        if pool_type == "cls":
            pooled_output = last_hidden_state[:, 0, :]
            pooled_output = self.pooler(pooled_output)
        elif pool_type == "mean":
            pooled_output = self.pooler(last_hidden_state, attention_mask)
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}")
            
        return pooled_output

def load_hf_model():
    hf_model = AutoModel.from_pretrained("roychowdhuryresearch/dna2vec", trust_remote_code=True)
    hf_tokenizer = AutoTokenizer.from_pretrained("roychowdhuryresearch/dna2vec", trust_remote_code=True)

    class AveragePooler(nn.Module):
        """
        Parameter-free poolers to get the sentence embedding
        # derived from https://github.com/princeton-nlp/SimCSE/blob/13361d0e29da1691e313a94f003e2ed1cfa97fef/simcse/models.py#LL49C1-L84C1
        """

        def __init__(self):
            super().__init__()

        def forward(self, last_hidden, attention_mask):
            # Old previous implementation
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
                -1
            ).unsqueeze(-1)

    hf_model.pooler = AveragePooler()
    return hf_model, hf_tokenizer, hf_model.pooler

def sft_model_from_config(cfg: SFTModelConfigSchema) -> Tuple[SFTModel, BPTokenizer]:
    """
    Create SFT model from configuration
    
    Args:
        cfg: SFT model configuration
        
    Returns:
        Tuple of (SFT model, tokenizer)
    """
    # Create base model components
    encoder, tokenizer, pooler = load_hf_model()
    
    # Create MLP head
    mlp_head = MLPHead(cfg.mlp_config)
    
    # Create SFT model
    sft_model = SFTModel(
        encoder=encoder,
        pooler=pooler,
        mlp_head=mlp_head,
        freeze_encoder=cfg.freeze_encoder
    )
    
    return sft_model, tokenizer


def load_pretrained_encoder(encoder: Encoder, checkpoint_path: Path) -> Encoder:
    """
    Load pre-trained encoder weights
    
    Args:
        encoder: Encoder instance
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Encoder with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    return encoder 