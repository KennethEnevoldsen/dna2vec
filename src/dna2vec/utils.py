import importlib
import importlib.util
import json
from pathlib import Path

from pydantic import BaseModel

from dna2vec.config_schema import ConfigSchema


def get_config_from_path(config_path: Path) -> ConfigSchema:
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(module)  # type: ignore

    CONFIG = module.CONFIG
    return CONFIG


def cfg_to_wandb_dict(cfg: BaseModel) -> dict:
    """
    Convert a ConfigSchema instance to a dictionary that can be logged to wandb by
    recursively unnesting the dict structure and joining then with a dot.

    E.g. {"model_config": {"embedding_dim": 384}} -> {"model_config.embedding_dim": 384}
    if it is a callable, the name of the callable is used instead of the value

    Args:
        cfg: ConfigSchema
    """
    cfg_dict = cfg.dict()

    def _cfg_to_wandb_dict(cfg_dict):
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                for k, v in _cfg_to_wandb_dict(value):
                    yield f"{key}.{k}", v
            elif callable(value):
                try:
                    name = value.__name__
                except:
                    name = value.__class__.__name__

                yield key, name
            else:
                # check if value if json serializable
                try:
                    json.dumps(value)
                except TypeError:
                    # if not, convert to string
                    value = str(value)
                yield key, value

    return dict(_cfg_to_wandb_dict(cfg_dict))


