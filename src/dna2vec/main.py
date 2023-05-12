"""
cli interface for training the contrastive self-supervised sequence model
"""


from pathlib import Path
from typing import Optional

import typer
import wandb
from torch.utils.data import DataLoader

from dna2vec.dataset import create_collate_fn
from dna2vec.model import Encoder
from dna2vec.trainer import ContrastiveTrainer
from dna2vec.utils import cfg_to_wandb_dict, get_config_from_path


def main(config_path: Optional[str] = None):
    """
    Args:
        cfg: Path to the configuration file, should be a .py file containing a Config instance
    """
    if config_path is None:
        cfg = Path(__file__).parents[2] / "configs" / "default_config.py"
    else:
        cfg = Path(config_path)

    config = get_config_from_path(cfg)
    # log config to wandb
    wandb.init(project="dna2vec", config=cfg_to_wandb_dict(config))

    model_cfg = config.model_config
    training_cfg = config.training_config
    dataset_cfg = config.dataset_config

    # MODEL: Model and tokenizer
    model = Encoder(**model_cfg.dict())
    collate_fn = create_collate_fn(tokenizer_path=model_cfg.tokenizer_path)

    # DATASET: Dataset creation
    dataset = dataset_cfg.dataset(**dataset_cfg.dict())

    # TRAINING: Optimizer, data loader and trainer
    optimizer_cfg = training_cfg.optimizer_config
    optimizer = training_cfg.optimizer(model.parameters(), **optimizer_cfg.dict())
    dataloader = DataLoader(
        dataset, batch_size=training_cfg.batch_size, collate_fn=collate_fn
    )

    trainer = ContrastiveTrainer(
        model=model,
        criterion=training_cfg.criterion,
        optimizer=optimizer,
        train_dataloader=dataloader,
        device=training_cfg.device,
    )

    # TRAINING: Training loop
    trainer.train(
        max_steps=training_cfg.max_steps,
        log_interval=training_cfg.log_interval,
    )


if __name__ == "__main__":
    typer.run(main)
