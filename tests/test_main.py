from dna2vec.config_schema import ConfigSchema, TrainingConfigSchema
from dna2vec.main import main


def test_main():
    CONFIG = ConfigSchema(
        training_config=TrainingConfigSchema(max_steps=4, batch_size=4)
    )
    main(CONFIG, wandb_mode="dryrun")
