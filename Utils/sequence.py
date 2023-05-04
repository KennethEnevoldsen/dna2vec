from typing import Literal, List, Any

import numpy as np
np.random.seed(42)



class Sequence():
    
    
    def __init__(self, sequence: str = "", limit: int = 5) -> None:
        if len(sequence) <= limit:
            raise ValueError("Sequence is of limited length.")
        sequence = sequence.replace("\n","") # incase newline characters exist
        self.sequence = sequence

    
    def split(self, mode: Literal["random", "fixed"] = "random", 
               sample_length: Any = None, number_of_sequences: int = 5) -> None:
        
        subsequences: List = []
        
        if mode == "random":
            
            if type(sample_length) != list:
                raise ValueError("Sample length is a 2-list of (min length, max_length)")
            
            for i in range(number_of_sequences):
                length = np.random.randint(sample_length[0], sample_length[1])
                start = np.random.randint(0, len(self.sequence) - length)
                end = start + length
                subsequences.append(self.sequence[start:end])
                

        elif mode == "fixed":
            
            if type(sample_length) != int:
                raise ValueError("Sample length is not an integer")
            
            elif sample_length > len(self.sequence):
                raise ValueError("Sample length is greater than the sequence length.")
            
            for i in range(number_of_sequences):
                start = np.random.randint(0, len(self.sequence) - sample_length)
                end = start + sample_length
                subsequences.append(self.sequence[start:end])
            
        else:
            raise ValueError("Mode is undefined. Please use: random, fixed.")

        return subsequences



if __name__ == "__main__":
    
    with open("sample", "r") as f:
        sequence = f.read()
        
    sequence = Sequence(sequence)
    subsequences = sequence.split(mode = "random", sample_length=[5,30], number_of_sequences=500)
    
    with open("subsequences_sample", "w+") as f:
        for seq in subsequences:
            f.write(seq)
            f.write("\n")
        