from pathlib import Path

import torch

from dna2vec.config_schema import (ConfigSchema,
                                   DatasetConfigSchemaUniformSampling,
                                   SchedulerConfigSchema, TrainingConfigSchema)
from dna2vec.dataset import FastaUniformSampler
from dna2vec.main import main

device = torch.device("cuda:4")
CONFIG = ConfigSchema(
    training_config=TrainingConfigSchema(
        max_steps=100_000,
        batch_size=32,
        device=device,
        log_interval=100,
        accumulation_steps=8,
        scheduler_config=SchedulerConfigSchema(
            max_lr=1e-4,
        ),
    ),
    # dataset_config=DatasetConfigSchema(
    #     fasta_file=Path("data/NC_000002.12.txt"),
    #     range_mean=2000,
    #     range_std=100,
    #     subsequence_range_mean=200,
    #     subsequence_range_std=20,
    #     sampling_strategy="random_subsequence",
    # ),
    dataset_config=DatasetConfigSchemaUniformSampling(fasta_file = [Path("data/NC_000002.12.txt"), Path("data/NC_000003.12.fasta")], dataset=FastaUniformSampler),
)


main(CONFIG, watch_watch=True)
