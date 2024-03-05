from pathlib import Path

import torch

from dna2vec.config_schema import (
    ConfigSchema,
    DatasetConfigSchemaUniformSampling,
    SchedulerConfigSchema,
    TrainingConfigSchema,
)
from dna2vec.dataset import FastaUniformSampler
from dna2vec.main import main

device = torch.device("cuda:3")
CONFIG = ConfigSchema(
    training_config=TrainingConfigSchema(
        max_steps=100_000,
        batch_size=16,
        device=device,
        log_interval=100,
        accumulation_steps=16,
        scheduler_config=SchedulerConfigSchema(
            max_lr=1e-4,
        ),
        regularizer=0,
        pool_type="cls",
    ),
    dataset_config=DatasetConfigSchemaUniformSampling(
        fasta_file=[Path("/mnt/SSD1/shreyas/dna2vec/data/chm13v2.0.fa")],
        range_min=800,
        range_max=2000,
        subsequence_range_min=80,
        subsequence_range_max=250,
        dataset=FastaUniformSampler,
        sampling_strategy="random_subsequence_uppercase",
        read_regularizer=True,
    ),
)


main(CONFIG, wandb_watch=True)
