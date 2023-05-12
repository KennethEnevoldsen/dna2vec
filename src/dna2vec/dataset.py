from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, IterableDataset

from dna2vec.tokenizer import CustomTokenizer


class FastaSamplerDataset(IterableDataset):
    def __init__(
        self,
        range_mean: float,
        range_std: float,
        fasta_file: Path,
    ):
        super().__init__()
        self.range_mean = range_mean
        self.range_std = range_std
        self.fasta_file = fasta_file

        # load in text file
        with open(self.fasta_file, "r") as f:
            self.text = f.read()

        self.len_text = len(self.text)

    def __iter__(self):
        """
        Randomly samples two sequence from the fasta file which constitute a positive
        sample.

        1) sample a random length L_1, and random sequence index i_1 as well as a direction [left, right]
        2) sample sequence x_1 from the range [i_1, i_1 +/- L_1]
        2) From the range of the first sequence sample a index
        3) the sample a direction (left or right) for the second sequence
        4) Sample a random length for the second sequence
        5) then sample the second sequence from the range of the first sequence
        """

        while True:
            i_1 = torch.randint(0, self.len_text, (1,))
            L_1 = torch.normal(self.range_mean, self.range_std, (1,)).int()
            # sample a direction
            direction_1 = torch.randint(0, 2, (1,))
            # sample a sequence
            if direction_1 == 0:
                x_1 = self.text[i_1 : i_1 + L_1]
            else:
                x_1 = self.text[i_1 - L_1 : i_1]

            # sample a second index
            i_2 = torch.randint(0, self.len_text, (1,))
            direction_2 = torch.randint(0, 2, (1,))
            L_2 = torch.normal(self.range_mean, self.range_std, (1,)).int()
            # sample a sequence
            if direction_2 == 0:
                x_2 = self.text[i_2 : i_2 + L_2]
            else:
                x_2 = self.text[i_2 - L_2 : i_2]

            yield x_1, x_2


def create_collate_fn(tokenizer_path: Path):
    tokenizer = CustomTokenizer.load(str(tokenizer_path))

    def collate_fn(batch) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        collate to max batch size and output a dictionary with two elements
        ids = matrix of shape (batch_size, max_sequence_length)
        attention_mask = matrix of shape (batch_size, max_sequence_length)
        """
        x_1, x_2 = list(zip(*batch))
        x_1 = tokenizer.tokenize(x_1)
        x_2 = tokenizer.tokenize(x_2)

        return x_1.to_torch(), x_2.to_torch()

    return collate_fn



