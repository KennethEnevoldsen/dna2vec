from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader # type: ignore

from dna2vec.dataset import FastaSamplerDataset, collate_fn
from dna2vec.tokenizer import BPTokenizer


def test_fasta_sampler_dataset():
    # test that it works
    project_path = Path(__file__).parent.parent
    fasta_file = project_path / "tests" / "data" / "NC_000002.12.txt"

    dataset = FastaSamplerDataset(range_mean=100, range_std=10, fasta_file=fasta_file)
    dataloader = DataLoader(dataset, batch_size=2)

    first_batch = next(iter(dataloader))

    assert len(first_batch) == 2 # type: ignore
    assert len(first_batch[0]) == 2 # type: ignore
    assert isinstance(first_batch[0][0], str)  # type: ignore


def test_collate_fn():
    project_path = Path(__file__).parent.parent
    fasta_file = project_path / "tests" / "data" / "NC_000002.12.txt"
    tokenizer_path = (
        project_path / "src" / "model" / "tokenizers" / "dna_tokenizer_10k.json"
    )

    dataset = FastaSamplerDataset(range_mean=100, range_std=10, fasta_file=fasta_file)
    tokenizer = BPTokenizer.load(str(tokenizer_path))
    _collate_fn = partial(collate_fn, tokenizer=tokenizer)

    dataloader = DataLoader(dataset, batch_size=4, collate_fn=_collate_fn)

    x_1, x_2 = next(iter(dataloader)) # type: ignore

    assert isinstance(x_1, dict)
    assert "input_ids" in x_1
    assert "attention_mask" in x_1
    assert x_1["input_ids"].shape[0] == 4