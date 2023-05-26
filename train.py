from pathlib import Path
from dna2vec.config_schema import (
    ConfigSchema,
    DatasetConfigSchema,
    TrainingConfigSchema,
    SchedulerConfigSchema,
)
from dna2vec.main import main
import torch


device = torch.device("cuda:3")
CONFIG = ConfigSchema(
    training_config=TrainingConfigSchema(
        max_steps=100_000,
        batch_size=64,
        device=device,
        log_interval=100,
        accumulation_steps=1,
        scheduler_config=SchedulerConfigSchema(total_steps=100), # otherwise reloading from a checkpoint will fail
    ),
    dataset_config=DatasetConfigSchema(
        fasta_file=Path("data/NC_000002.12.txt"),
        range_mean=1000,
        range_std=100,
        subsequence_range_mean=200,
        subsequence_range_std=20,
        sampling_strategy="random_subsequence",
    ),
)


main(CONFIG, watch_watch=True)
