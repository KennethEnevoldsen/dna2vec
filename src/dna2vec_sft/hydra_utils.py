"""
Utilities for Hydra configuration management in DNA2Vec SFT
"""

from pathlib import Path
from typing import Dict, List, Optional
import torch
from omegaconf import DictConfig, OmegaConf



def auto_detect_device() -> str:
    """Auto-detect the best available device"""
    if torch.cuda.is_available():
        # Use the first available GPU
        return f"cuda:0"
    else:
        return "cpu"


def validate_paths(cfg: DictConfig) -> None:
    """
    Validate that all required paths exist
    
    Args:
        cfg: Hydra configuration object
        
    Raises:
        FileNotFoundError: If required files don't exist
    """
    # Check model checkpoint path
    if cfg.model_config.pretrained_encoder_path:
        path = Path(cfg.model_config.pretrained_encoder_path)
        if not path.exists():
            raise FileNotFoundError(f"Pretrained encoder not found: {path}")
    
    # Check training dataset files
    for fasta_path in cfg.dataset_config.fasta_file:
        path = Path(fasta_path)
        if not path.exists():
            raise FileNotFoundError(f"Training FASTA file not found: {path}")
    
    # Check validation dataset files if specified
    if cfg.val_dataset_config:
        for fasta_path in cfg.val_dataset_config.fasta_file:
            path = Path(fasta_path)
            if not path.exists():
                raise FileNotFoundError(f"Validation FASTA file not found: {path}")


def print_config_summary(cfg: DictConfig) -> None:
    """
    Print a readable summary of the configuration
    
    Args:
        cfg: Hydra configuration object
    """
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    
    # Experiment info
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Run Name: {cfg.run_name}")
    
    # Model config
    print(f"\nModel Configuration:")
    print(f"  Embedding Dim: {cfg.model_config.embedding_dim}")
    print(f"  MLP Hidden Dim: {cfg.model_config.mlp_config.hidden_dim}")
    print(f"  MLP Output Dim: {cfg.model_config.mlp_config.output_dim}")
    print(f"  MLP Layers: {cfg.model_config.mlp_config.num_layers}")
    print(f"  Freeze Encoder: {cfg.model_config.freeze_encoder}")
    
    # Training config
    print(f"\nTraining Configuration:")
    print(f"  Max Steps: {cfg.training_config.max_steps}")
    print(f"  Batch Size: {cfg.training_config.batch_size}")
    print(f"  Learning Rate: {cfg.training_config.optimizer_config.lr}")
    print(f"  Device: {cfg.training_config.device}")
    print(f"  Pool Type: {cfg.training_config.pool_type}")
    
    # Dataset config
    print(f"\nDataset Configuration:")
    print(f"  Strategy: {cfg.dataset_config.sampling_strategy}")
    print(f"  Fragment Range: {cfg.dataset_config.range_min}-{cfg.dataset_config.range_max}")
    print(f"  Read Range: {cfg.dataset_config.subsequence_range_min}-{cfg.dataset_config.subsequence_range_max}")
    print(f"  FASTA Files: {len(cfg.dataset_config.fasta_file)} file(s)")
    
    # WandB config
    print(f"\nLogging Configuration:")
    print(f"  WandB Mode: {cfg.wandb.mode}")
    print(f"  Project: {cfg.wandb.project}")
    
    print("="*60 + "\n")


def create_experiment_variants(base_cfg: DictConfig) -> Dict[str, DictConfig]:
    """
    Create different experimental variants from a base configuration
    
    Args:
        base_cfg: Base Hydra configuration
        
    Returns:
        Dictionary of experiment variants
    """
    variants = {}
    
    # Create a quick test variant
    test_cfg = OmegaConf.copy(base_cfg)
    test_cfg.training_config.max_steps = 1000
    test_cfg.training_config.batch_size = 8
    test_cfg.run_name = f"test_{test_cfg.run_name}"
    variants["test"] = test_cfg
    
    # Create a high learning rate variant
    high_lr_cfg = OmegaConf.copy(base_cfg)
    high_lr_cfg.training_config.optimizer_config.lr = 5e-5
    high_lr_cfg.training_config.scheduler_config.max_lr = 5e-5
    high_lr_cfg.run_name = f"high_lr_{high_lr_cfg.run_name}"
    variants["high_lr"] = high_lr_cfg
    
    # Create a large model variant
    large_cfg = OmegaConf.copy(base_cfg)
    large_cfg.model_config.mlp_config.hidden_dim = 1024
    large_cfg.model_config.mlp_config.num_layers = 3
    large_cfg.run_name = f"large_{large_cfg.run_name}"
    variants["large"] = large_cfg
    
    # Create a frozen encoder variant
    frozen_cfg = OmegaConf.copy(base_cfg)
    frozen_cfg.model_config.freeze_encoder = True
    frozen_cfg.training_config.optimizer_config.lr = 1e-4  # Higher LR for frozen encoder
    frozen_cfg.run_name = f"frozen_{frozen_cfg.run_name}"
    variants["frozen"] = frozen_cfg
    
    return variants


def save_config_to_file(cfg: DictConfig, filepath: Path) -> None:
    """
    Save configuration to a YAML file
    
    Args:
        cfg: Configuration to save
        filepath: Where to save the config
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, filepath)


def load_config_from_file(filepath: Path) -> DictConfig:
    """
    Load configuration from a YAML file
    
    Args:
        filepath: Path to the config file
        
    Returns:
        Loaded configuration
    """
    return OmegaConf.load(filepath)


class ConfigManager:
    """
    Helper class for managing multiple configurations
    """
    
    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self.experiments = {}
    
    def register_experiment(self, name: str, cfg: DictConfig) -> None:
        """Register an experiment configuration"""
        self.experiments[name] = cfg
    
    def get_experiment(self, name: str) -> DictConfig:
        """Get an experiment configuration by name"""
        if name not in self.experiments:
            raise ValueError(f"Experiment '{name}' not found. Available: {list(self.experiments.keys())}")
        return self.experiments[name]
    
    def list_experiments(self) -> List[str]:
        """List all registered experiments"""
        return list(self.experiments.keys())
    
    def save_experiment(self, name: str, cfg: Optional[DictConfig] = None) -> None:
        """Save an experiment configuration to disk"""
        if cfg is None:
            cfg = self.get_experiment(name)
        
        save_path = self.config_dir / f"{name}.yaml"
        save_config_to_file(cfg, save_path)
        print(f"Saved experiment '{name}' to {save_path}")
    
    def load_experiment(self, name: str) -> DictConfig:
        """Load an experiment configuration from disk"""
        load_path = self.config_dir / f"{name}.yaml"
        cfg = load_config_from_file(load_path)
        self.register_experiment(name, cfg)
        return cfg 