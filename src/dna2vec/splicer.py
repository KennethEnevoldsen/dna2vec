"""
Consider rewriting in generator form.
It doesn't appear that we need it though.
Bulk is a list of strings.
"""

from typing import Any, List, Literal
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser('datapath', description="Path to I/O data")
parser = argparse.add_argument('mode', description="Splicing Mode")
parser = argparse.add_argument('n', description="Number of splices")

args = parser.parse_args()
np.random.seed(42)


class Splicer:
    def __init__(self, sequence: str = "", limit: int = 5) -> None:
        if len(sequence) <= limit:
            raise ValueError("Sequence is of limited length.")
        sequence = sequence.replace("\n", "")  # incase newline characters exist
        self.sequence = sequence

    def splice(
        self,
        mode: Literal["random", "fixed"] = "random",
        sample_length: Any = None,
        number_of_sequences: int = 5,
    ) -> None:
        subsequences: List = []

        if mode == "random":
            if type(sample_length) != list:
                raise ValueError(
                    "Sample length is a 2-list of (min length, max_length)"
                )

            for _ in tqdm(range(number_of_sequences)):
                length = np.random.randint(sample_length[0], sample_length[1])
                start = np.random.randint(0, len(self.sequence) - length)
                end = start + length
                subsequences.append(self.sequence[start:end])

        elif mode == "fixed":
            if type(sample_length) != int:
                raise ValueError("Sample length is not an integer")

            if sample_length > len(self.sequence):
                raise ValueError("Sample length is greater than the sequence length.")

            for _ in tqdm(range(number_of_sequences)):
                start = np.random.randint(0, len(self.sequence) - sample_length)
                end = start + sample_length
                subsequences.append(self.sequence[start:end])

        else:
            raise ValueError("Mode is undefined. Please use: random, fixed.")

        return subsequences


if __name__ == "__main__":
    
    import os
    
    data_path = args["data_path"]
    with open(os.path.join(data_path, "sample.txt"), "r") as f:
        sequence = f.read()

    sequence = Splicer(sequence)
    subsequences = sequence.splice(
        mode=args["mode"], sample_length=[5, 30], number_of_sequences=args["n"]
    )

    with open(os.path.join(data_path, "subsequences_sample.txt"), "w+") as f:
        for seq in subsequences:
            f.write(seq)
            f.write("\n")
    
