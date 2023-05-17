from dna2vec.config_schema import ConfigSchema, TrainingConfigSchema
from dna2vec.main import main
from pathlib import Path
import warnings
import torch
from dna2vec.trainer import ContrastiveTrainer

def test_main():

    test_folder = Path(__file__).parent
    CONFIG = ConfigSchema(
        training_config=TrainingConfigSchema(max_steps=10, batch_size=4, log_interval=1, save_path=test_folder/"test_models")
    )
    main(CONFIG, wandb_mode="dryrun")

    # test that we can load from checkpoint
    save_path = test_folder/"test_models"/"checkpoint.pt"

    trainer = ContrastiveTrainer.load_from_disk(str(save_path))

    assert trainer.config.training_config.max_steps == 10
    trainer.train(max_steps=2)




    

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

