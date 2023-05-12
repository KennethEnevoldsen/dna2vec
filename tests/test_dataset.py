from pathlib import Path

from torch.utils.data import DataLoader

from dna2vec.dataset import FastaSamplerDataset, create_collate_fn


def test_fasta_sampler_dataset():
    # test that it works
    project_path = Path(__file__).parent.parent
    fasta_file = project_path / "tests" / "data" / "NC_000002.12.txt"

    dataset = FastaSamplerDataset(range_mean=100, range_std=10, fasta_file=fasta_file)
    dataloader = DataLoader(dataset, batch_size=2)

    first_batch = next(iter(dataloader))

    assert len(first_batch) == 2
    assert len(first_batch[0]) == 2
    assert isinstance(first_batch[0][0], str)


def test_collate_fn():
    project_path = Path(__file__).parent.parent
    fasta_file = project_path / "tests" / "data" / "NC_000002.12.txt"
    tokenizer_path = (
        project_path / "src" / "model" / "tokenizers" / "dna_tokenizer_10k.json"
    )

    dataset = FastaSamplerDataset(range_mean=100, range_std=10, fasta_file=fasta_file)
    collate_fn = create_collate_fn(tokenizer_path)

    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    x_1, x_2 = next(iter(dataloader))

    assert isinstance(x_1, dict)
    assert "input_ids" in x_1
    assert "attention_mask" in x_1
    assert x_1["input_ids"].shape[0] == 4
