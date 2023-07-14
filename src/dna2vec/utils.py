import importlib
import importlib.util
import json
import os
import shutil
import urllib.request
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm

from dna2vec.config_schema import ConfigSchema

CACHE_DIR = Path.home() / ".cache" / "dna2vec"


def get_human_reference_genome(force: bool = False) -> Path:
    """
    Download the human reference genome to the cache directory. If the file already
    exists, it will not be downloaded again unless force is set to True.

    Args:
        force: If True, the file will be downloaded even if it already exists.
    """
    cache_dir = get_cache_dir()
    cache_dir.mkdir(exist_ok=True, parents=True)

    url = "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/GRCh38_latest_genomic.fna.gz"
    output = cache_dir / "GRCh38_latest_genomic.fna"

    if not output.exists() or force:
        download_url(url, output)
        shutil.unpack_archive(str(output), extract_dir=str(cache_dir))
        os.remove(output)

    return cache_dir / "GRCh38_latest_genomic.fna"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def get_cache_dir() -> Path:
    """
    Get the cache directory for dna2vec. Can be overridden by setting the environment
    variable DNA2VEC_CACHE_DIR.
    """
    cache_dir = os.environ.get("DNA2VEC_CACHE_DIR")
    if cache_dir is not None:
        return Path(cache_dir)
    return CACHE_DIR


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
