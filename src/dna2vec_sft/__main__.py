"""
Command line interface for DNA2Vec SFT training

Usage:
    python -m dna2vec_sft
    python -m dna2vec_sft --config path/to/config.py
    python -m dna2vec_sft --quick --mlp-output-dim 128 --max-steps 25000
"""

import argparse
import sys
from pathlib import Path

import torch

from .sft_main import sft_main
from .train_sft import CONFIG as DEFAULT_CONFIG
from . import quick_sft_setup


def parse_args():
    parser = argparse.ArgumentParser(
        description="DNA2Vec SFT Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration options
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file (.py file with CONFIG variable)"
    )
    
    # Quick setup options
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Use quick setup with command line arguments"
    )
    
    # Model parameters
    parser.add_argument(
        "--embedding-dim", 
        type=int, 
        default=1020, 
        help="Encoder embedding dimension"
    )
    parser.add_argument(
        "--mlp-hidden-dim", 
        type=int, 
        default=512, 
        help="MLP hidden layer dimension"
    )
    parser.add_argument(
        "--mlp-output-dim", 
        type=int, 
        default=256, 
        help="MLP output dimension"
    )
    parser.add_argument(
        "--freeze-encoder", 
        action="store_true", 
        help="Freeze encoder weights during training"
    )
    parser.add_argument(
        "--pretrained-encoder", 
        type=str, 
        help="Path to pretrained encoder checkpoint"
    )
    
    # Training parameters
    parser.add_argument(
        "--max-steps", 
        type=int, 
        default=50_000, 
        help="Maximum training steps"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=16, 
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=5e-5, 
        help="Learning rate"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0", 
        help="Device to use for training"
    )
    
    # Data parameters
    parser.add_argument(
        "--fasta-file", 
        type=str, 
        default="/mnt/SSD1/shreyas/dna2vec/data/chromosome_2/NC_000002.fasta",
        help="Path to FASTA file"
    )
    
    # Logging parameters
    parser.add_argument(
        "--wandb-mode", 
        type=str, 
        default="online", 
        choices=["online", "offline", "disabled"],
        help="WandB logging mode"
    )
    parser.add_argument(
        "--no-wandb-watch", 
        action="store_true", 
        help="Disable WandB model watching"
    )
    
    return parser.parse_args()


def load_config_from_file(config_path: str):
    """Load configuration from a Python file"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Import the config file as a module
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    if not hasattr(config_module, 'CONFIG'):
        raise AttributeError(f"Config file {config_path} must contain a CONFIG variable")
    
    return config_module.CONFIG


def main():
    """Main CLI entry point"""
    args = parse_args()
    
    # Determine configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config_from_file(args.config)
    elif args.quick:
        print("Using quick setup with command line arguments")
        config = quick_sft_setup(
            embedding_dim=args.embedding_dim,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_output_dim=args.mlp_output_dim,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            fasta_file=args.fasta_file,
            freeze_encoder=args.freeze_encoder,
            pretrained_encoder_path=args.pretrained_encoder,
        )
    else:
        print("Using default configuration")
        config = DEFAULT_CONFIG
        
        # Apply any command line overrides to default config
        if args.max_steps != 50_000:
            config.training_config.max_steps = args.max_steps
        if args.batch_size != 16:
            config.training_config.batch_size = args.batch_size
        if args.learning_rate != 5e-5:
            config.training_config.scheduler_config.max_lr = args.learning_rate
        if args.device != "cuda:0":
            config.training_config.device = torch.device(args.device)
        if args.mlp_output_dim != 256:
            config.model_config.mlp_config.output_dim = args.mlp_output_dim
        if args.freeze_encoder:
            config.model_config.freeze_encoder = True
        if args.pretrained_encoder:
            config.model_config.pretrained_encoder_path = Path(args.pretrained_encoder)
    
    # Print configuration summary
    print("\n=== SFT Training Configuration ===")
    print(f"Device: {config.training_config.device}")
    print(f"Max steps: {config.training_config.max_steps}")
    print(f"Batch size: {config.training_config.batch_size}")
    print(f"Learning rate: {config.training_config.scheduler_config.max_lr}")
    print(f"MLP output dim: {config.model_config.mlp_config.output_dim}")
    print(f"Encoder frozen: {config.model_config.freeze_encoder}")
    if config.model_config.pretrained_encoder_path:
        print(f"Pretrained encoder: {config.model_config.pretrained_encoder_path}")
    print("===================================\n")
    
    # Start training
    try:
        sft_main(
            config, 
            wandb_mode=args.wandb_mode, 
            wandb_watch=not args.no_wandb_watch
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 