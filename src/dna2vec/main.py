"""
cli interface for training the contrastive self-supervised sequence model
"""


from functools import partial
from pathlib import Path
from typing import Optional

import typer
from torch.utils.data import DataLoader

import wandb
from dna2vec.config_schema import ConfigSchema
from dna2vec.dataset import collate_fn
from dna2vec.model import Encoder
from dna2vec.tokenizer import CustomTokenizer
from dna2vec.trainer import ContrastiveTrainer
from dna2vec.utils import cfg_to_wandb_dict, get_config_from_path


def main(config: ConfigSchema, wandb_mode: str = "online"):
    """
    Args:
        config: The configuration object
    """
    model_cfg = config.model_config
    training_cfg = config.training_config
    dataset_cfg = config.dataset_config

    # MODEL: Model and tokenizer
    model_kwargs = model_cfg.dict()
    pooler = model_kwargs.pop("pooling")
    tokenizer_path = model_kwargs.pop("tokenizer_path")
    tokenizer = CustomTokenizer.load(str(tokenizer_path))

    if model_cfg.vocab_size is None:
        model_kwargs["vocab_size"] = tokenizer.vocab_size + 1
        model_cfg.vocab_size = tokenizer.vocab_size + 1

    model = Encoder(**model_kwargs)
    _collate_fn = partial(collate_fn, tokenizer=tokenizer)

    # DATASET: Dataset creation
    dataset_kwargs = dataset_cfg.dict()
    dataset_fn = dataset_kwargs.pop("dataset")
    dataset = dataset_fn(**dataset_kwargs)

    # TRAINING: Optimizer, data loader and trainer
    optimizer_cfg = training_cfg.optimizer_config
    optimizer = training_cfg.optimizer(model.parameters(), **optimizer_cfg.dict())
    dataloader = DataLoader(
        dataset, batch_size=training_cfg.batch_size, collate_fn=_collate_fn
    )
    sim = training_cfg.similarity(temperature=training_cfg.temperature)

    trainer = ContrastiveTrainer(
        encoder=model,
        pooling=pooler,
        loss=training_cfg.loss,
        optimizer=optimizer,
        train_dataloader=dataloader,
        device=training_cfg.device,
        similarity=sim,
    )

    # log config to wandb
    wandb.init(project="dna2vec", config=cfg_to_wandb_dict(config), mode=wandb_mode)

    # TRAINING: Training loop
    trainer.train(
        max_steps=training_cfg.max_steps,
        log_interval=training_cfg.log_interval,
    )


def main_cli(config_path: Optional[str] = None):
    """
    The main cli interface for training the model

    Args:
        cfg: Path to the configuration file, should be a .py file containing a Config instance
    """
    if config_path is None:
        cfg = Path(__file__).parents[2] / "configs" / "default_config.py"
    else:
        cfg = Path(config_path)

    config = get_config_from_path(cfg)
    main(config)


if __name__ == "__main__":
    typer.run(main_cli)
