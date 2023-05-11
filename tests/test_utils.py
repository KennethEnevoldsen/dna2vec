from pathlib import Path

from dna2vec.utils import cfg_to_wandb_dict, get_config_from_path


def test_cfg_to_wandb_dict():
    cfg = get_config_from_path(
        Path(__file__).parents[1] / "configs" / "default_config.py"
    )
    print(cfg)
    wandb_dict = cfg_to_wandb_dict(cfg)
    # check that it is json serializable
    import json

    print(wandb_dict)
    json.dumps(wandb_dict)
