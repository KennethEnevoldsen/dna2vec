from pathlib import Path
from dna2vec.config_schema import ConfigSchema, DatasetConfigSchema, TrainingConfigSchema
from dna2vec.main import main
import torch


device = torch.device("cuda:4")
CONFIG = ConfigSchema(training_config=TrainingConfigSchema(max_steps=100_000, batch_size=256, device=device, log_interval=100, accumulation_steps=4),
                      dataset_config=DatasetConfigSchema(fasta_file=Path("data/NC_000002.12.txt"))
                      )


main(CONFIG, watch_watch=True)