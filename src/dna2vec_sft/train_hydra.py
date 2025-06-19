"""
Hydra-enabled SFT training script for DNA2Vec
"""

import os
from pathlib import Path
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import importlib

from dna2vec_sft.sft_config_schema import SFTConfigSchema
from dna2vec_sft.sft_main import sft_main


def get_class(class_path: str):
    """Helper to import class from string path."""
    # Check if the class path is valid
    if not isinstance(class_path, str) or '.' not in class_path:
        raise ValueError(f"Invalid class path: {class_path}")
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def hydra_cfg_to_pydantic(cfg: DictConfig) -> SFTConfigSchema:
    """
    Convert Hydra config to Pydantic config object with proper validation
    by manually converting types.
    """
    # Convert DictConfig to a plain python dict, resolving interpolations
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # --- Perform manual type conversions to match Pydantic schema ---

    # training_config conversions
    if 'training_config' in cfg_dict:
        tc = cfg_dict['training_config']
        if 'optimizer' in tc:
            tc['optimizer'] = get_class(tc['optimizer'])
        if 'scheduler' in tc:
            tc['scheduler'] = get_class(tc['scheduler'])
        if 'similarity' in tc:
            tc['similarity'] = get_class(tc['similarity'])
        if 'loss' in tc:
            tc['loss'] = get_class(tc['loss'])()  # Instantiate the loss object
        
        device_str = tc.get('device', 'auto')
        if device_str == "auto":
            tc['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            tc['device'] = torch.device(device_str)

    # dataset_config conversions
    if 'dataset_config' in cfg_dict:
        dc = cfg_dict['dataset_config']
        if 'dataset' in dc:
            dc['dataset'] = get_class(dc['dataset'])
        if 'fasta_file' in dc:
            dc['fasta_file'] = [Path(dc['fasta_file'])]

    # val_dataset_config conversions
    if 'val_dataset_config' in cfg_dict and cfg_dict['val_dataset_config']:
        vdc = cfg_dict['val_dataset_config']
        if 'dataset' in vdc:
            vdc['dataset'] = get_class(vdc['dataset'])
        if 'fasta_file' in vdc:
            vdc['fasta_file'] = [Path(vdc['fasta_file'])]

    # model_config conversions
    if 'model_config' in cfg_dict and 'pretrained_encoder_path' in cfg_dict['model_config'] and cfg_dict['model_config']['pretrained_encoder_path']:
        cfg_dict['model_config']['pretrained_encoder_path'] = Path(cfg_dict['model_config']['pretrained_encoder_path'])

    # Clean up redundant *_config keys that the user added with _target_
    # These are not needed as the main keys are now correctly typed.
    if 'training_config' in cfg_dict:
        cfg_dict['training_config'].pop('optimizer_config', None)
        cfg_dict['training_config'].pop('scheduler_config', None)
        cfg_dict['training_config'].pop('loss_config', None)
        cfg_dict['training_config'].pop('similarity_config', None)
        cfg_dict['training_config'].pop('device_config', None)
    
    # Create the Pydantic object now that types are correct
    config = SFTConfigSchema(**cfg_dict)
    
    return config


@hydra.main(version_base=None, config_path="configs", config_name="experiment/triplet_loss")
def main(cfg: DictConfig) -> None:
    """
    Main training function using Hydra configuration
    
    Args:
        cfg: Hydra configuration object
    """
    
    print("=== DNA2Vec SFT Training with Hydra ===")
    # Convert Hydra config to Pydantic for validation
    try:
        config = hydra_cfg_to_pydantic(cfg)
        print("✓ Configuration validation passed")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        raise
    
    # Set save path to Hydra's output directory
    config.training_config.save_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Run name: {cfg.run_name}")
    
    
    # Print key configuration details
    print(f"\nConfiguration Summary:")
    print(f"  Model: {config.model_config.mlp_config.output_dim}D output, {config.model_config.mlp_config.num_layers} layers")
    print(f"  Training: {config.training_config.max_steps} steps, batch size {config.training_config.batch_size}")
    print(f"  Dataset: {config.dataset_config.sampling_strategy}")
    print(f"  Device: {config.training_config.device}")
    print(f"  Pool type: {config.training_config.pool_type}")
    
    # Set up WandB with Hydra integration
    if cfg.wandb.mode != "disabled":
        # Override WandB config with Hydra values
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        
        # Initialize WandB
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=wandb_config,
            mode=cfg.wandb.mode,
            dir=os.getcwd(),  # Use Hydra's output directory
        )
        print(f"✓ WandB initialized: {cfg.wandb.project}/{cfg.wandb.name}")
    
    # Run training
    try:
        sft_main(
            config=config,
            wandb_mode=cfg.wandb.mode,
            wandb_watch=cfg.wandb.watch
        )
        print("✓ Training completed successfully")
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        if cfg.wandb.mode != "disabled":
            wandb.finish(exit_code=1)
        raise
    
    # Clean up WandB
    if cfg.wandb.mode != "disabled":
        wandb.finish()
    
    print(f"Outputs saved to: {os.getcwd()}")


if __name__ == "__main__":
    main() 