from typing import Any, List, Literal
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Path to I/O data")
parser.add_argument('--datapath', type=str)
parser.add_argument('--mode_train', type=str)
# parser.add_argument('--mode_test', type=str)
# parser.add_argument('--mode_test', type=str)
parser.add_argument('--rawfile', type=str)
parser.add_argument('--unit_length', type=int)
parser.add_argument('--meta', type=str)
parser.add_argument('--overlap', type=int)
parser.add_argument('--topath', type=str)
parser.add_argument('--ntrain', type=int)
# parser.add_argument('--ntest', type=int)


np.random.seed(42)


class Splicer:
    def __init__(self, 
                 sequence: str = "", 
                 limit: int = 5,
                 ) -> None:
        
        if len(sequence) <= limit:
            raise ValueError("Sequence is of limited length.")
        
        if ">" == sequence[0]: # FASTA File escape the first line
            _, sequence = sequence.split('\n', 1)
            
        sequence = sequence.replace("\n", "")  # incase newline characters exist
        self.sequence = sequence

    def splice(
        self,
        mode: Literal["random", "fixed", "hard_serialized"] = "random",
        sample_length: Any = None,
        overlap: Any = None,
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

        elif mode == "hard_serialized": # default
            if type(sample_length) != int:
                raise ValueError("Sample length is not an integer")

            if sample_length > len(self.sequence):
                raise ValueError("Sample length is greater than the sequence length.")
            
            start = 0
            sample_count = 0
            while start < len(self.sequence) and sample_count < 500000: # NASA-esque hard upper limit
                subsequences.append([
                    self.sequence[start:min(start + sample_length, len(self.sequence))], 
                    str(start)
                    ]
                )
                start += sample_length
                if overlap != None and overlap < sample_length:
                    start -= overlap
                else:
                    print("Overlap too large or not provided. Falling back to hard cutoffs.")
                sample_count += 1
                
            
        else:
            raise ValueError("Mode is undefined. Please use: random, fixed, hard_serialized.")

        return subsequences


if __name__ == "__main__":
    
    import os
    args = parser.parse_args()

    data_path = args.datapath
    raw_file = args.rawfile
    
    if ".fasta" not in raw_file:
        raise FileNotFoundError("Fasta file not found error.")

    meta_data = args.meta
    to_file = "floodfill.txt" if args.topath is None else args.topath
    
    with open(os.path.join(data_path, raw_file), "r") as f:
        sequence = f.read()

    sequence = Splicer(sequence)
    subsequences = sequence.splice(
        mode=args.mode_train, 
        sample_length=args.unit_length, 
        number_of_sequences=args.ntrain,
        overlap=args.overlap
    )

    import pickle
    global_dictionary = []
    for seq in tqdm(subsequences):
        global_dictionary.append({
            "text": seq[0],
            "position": seq[1],
            "metadata": meta_data
        })

    file = open(os.path.join(data_path, to_file), 'wb')
    pickle.dump(global_dictionary, file)
    file.close()

        
