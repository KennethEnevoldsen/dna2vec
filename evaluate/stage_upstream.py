from typing import Any, List, Literal
import numpy as np
from tqdm import tqdm
from helpers import read_fasta_chromosomes
from omegaconf import DictConfig, OmegaConf
import hydra
import pickle
import os

np.random.seed(42)

class Splicer:
    def __init__(
        self,
        sequence: str = "",
        limit: int = 5,
    ) -> None:

        if len(sequence) <= limit:
            raise ValueError("Sequence is of limited length.")

        if ">" == sequence[0]:  # FASTA File escape the first line
            _, sequence = sequence.split("\n", 1)

        sequence = sequence.replace("\n", "")  # incase newline characters exist
        self.sequence = sequence
        self.len_sequence = len(self.sequence)

    def generate_subsequences(
        self,
        mode: Literal["random", "fixed", "hard_serialized"] = "random",
        sample_length: Any = None,
        overlap: Any = None,
        starting_offset: int = None,
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

        elif mode == "hard_serialized":  # default
            if type(sample_length) != int:
                raise ValueError("Sample length is not an integer")

            if sample_length > len(self.sequence):
                raise ValueError("Sample length is greater than the sequence length.")

            start = 0
            sample_count = 0
            while (
                start < len(self.sequence) and sample_count < 10000000
            ):  # NASA-esque hard upper limit
                subsequences.append(
                    [
                        self.sequence[
                            start : min(start + sample_length, len(self.sequence))
                        ].upper(),
                        str(start + starting_offset),
                        str(start),
                    ]
                )
                start += sample_length
                if overlap != None and overlap < sample_length:
                    start -= overlap
                else:
                    print(
                        "Overlap too large or not provided. Falling back to hard cutoffs."
                    )
                sample_count += 1

        else:
            raise ValueError(
                "Mode is undefined. Please use: random, fixed, hard_serialized."
            )

        return subsequences
    
@hydra.main(config_path="configs", config_name="stage_upstream_config.yaml")
def process_fasta_and_store_subsequences(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    data_path = cfg.datapath
    raw_file = cfg.rawfile

    global_dictionary = []

    if ".fasta" not in raw_file and ".fa" not in raw_file:
        raise FileNotFoundError("Fasta file not found error.")

    # meta_data = args.meta
    to_file = "floodfill.txt" if cfg.topath is None else cfg.topath
    starting_offset = 0

    for header, sequence in read_fasta_chromosomes(os.path.join(data_path, raw_file)):
        sequence_obj = Splicer(sequence)
        subsequences = sequence_obj.generate_subsequences(
            mode=cfg.mode_train,
            sample_length=cfg.unit_length,
            number_of_sequences=cfg.ntrain,
            overlap=cfg.overlap,
            starting_offset=starting_offset,
        )
        starting_offset += sequence_obj.len_sequence

        for seq in tqdm(subsequences):
            global_dictionary.append(
                {
                    "text": seq[0],
                    "position": seq[1],
                    "local_position": seq[2],
                    "metadata": header,
                }
            )
        print(len(global_dictionary))
    file = open(os.path.join(data_path, to_file), "wb")
    pickle.dump(global_dictionary, file)
    file.close()


if __name__ == "__main__":

    process_fasta_and_store_subsequences()