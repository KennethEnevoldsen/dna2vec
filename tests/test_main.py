from dna2vec.config_schema import ConfigSchema, TrainingConfigSchema
from dna2vec.main import main

import warnings
import torch

def test_main():
    CONFIG = ConfigSchema(
        training_config=TrainingConfigSchema(max_steps=4, batch_size=4)
    )
    main(CONFIG, wandb_mode="dryrun")

def test_main_with_accelerator():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): # type: ignore
        device = torch.device("mps")
    else:
        warnings.warn("No accelerators found, skipping test")
        return

    CONFIG = ConfigSchema(
        training_config=TrainingConfigSchema(max_steps=4, batch_size=4, device=device)
    )
    main(CONFIG, wandb_mode="dryrun")
