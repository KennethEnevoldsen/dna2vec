"""
Consider rewriting in generator form.
It doesn't appear that we need it though.
Bulk is a list of strings.
"""

import argparse
from typing import Any, List, Literal

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Path to I/O data")
parser.add_argument('--datapath', type=str)
parser.add_argument('--mode_train', type=str)
parser.add_argument('--mode_test', type=str)
parser.add_argument('--ntrain', type=int)
parser.add_argument('--ntest', type=int)


np.random.seed(42)


class Splicer:
    def __init__(self, 
                 sequence: str = "", 
                 limit: int = 5,
                 ) -> None:
        
        if len(sequence) <= limit:
            raise ValueError("Sequence is of limited length.")
        sequence = sequence.replace("\n", "")  # incase newline characters exist
        self.sequence = sequence

    def splice(
        self,
        mode: Literal["random", "fixed", "hard_serialized"] = "random",
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
                subsequences.append([self.sequence[start:end], str(start)])

        elif mode == "fixed":
            if type(sample_length) != int:
                raise ValueError("Sample length is not an integer")

            if sample_length > len(self.sequence):
                raise ValueError("Sample length is greater than the sequence length.")

            for _ in tqdm(range(number_of_sequences)):
                start = np.random.randint(0, len(self.sequence) - sample_length)
                end = start + sample_length
                subsequences.append([self.sequence[start:end], str(start)])

        elif mode == "hard_serialized":
            if type(sample_length) != int:
                raise ValueError("Sample length is not an integer")

            if sample_length > len(self.sequence):
                raise ValueError("Sample length is greater than the sequence length.")
            
            start = 0
            sample_count = 0
            while start < len(self.sequence) and sample_count < 500000:
                subsequences.append([
                    self.sequence[start:min(start + sample_length, len(self.sequence))], 
                    str(start)
                    ]
                )
                # if sample_count < 2:
                #     print(subsequences)
                # else:
                #     exit()
                start += sample_length
                sample_count += 1
                
            
        else:
            raise ValueError("Mode is undefined. Please use: random, fixed.")

        return subsequences


if __name__ == "__main__":
    
    import os
    args = parser.parse_args()

    data_path = args.datapath
    with open(os.path.join(data_path, "NC_000002.12.txt"), "r") as f:
        sequence = f.read()

    sequence = Splicer(sequence)
    subsequences = sequence.splice(
        mode=args.mode_train, sample_length=1000, number_of_sequences=args.ntrain
    )
    print(subsequences[0])
    with open(os.path.join(data_path, "subsequences_sample_train.txt"), "w+") as f:
        for seq in tqdm(subsequences):
            f.write(" <> ".join(seq))
            f.write("\n")
    
    subsequences = sequence.splice(
        mode=args.mode_test, sample_length=[150, 250], number_of_sequences=args.ntest
    )

    with open(os.path.join(data_path, "subsequences_sample_test.txt"), "w+") as f:
        for seq in subsequences:
            f.write(" <> ".join(seq))
            f.write("\n")