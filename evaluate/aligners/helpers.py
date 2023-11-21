# Read the reference genome from a FASTA file
from tqdm import tqdm
from .smith_waterman import calculate_smith_waterman_distance
import numpy as np
from Bio import SeqIO

def load_reference_genome(file_path):
    # Initialize an empty dictionary to store chromosome sequences
    reference_dict = {}

    # Open the FASTA file and read the sequences
    with open(file_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            # Store each chromosome sequence in the dictionary
            reference_dict[record.id] = str(record.seq)

    return reference_dict

# Example SAM line
# sam_line = "chr1-500 0 chr1 49251123 60 250M * 0 0 CAAATTCCTTTCCCATAAAGCCTCCTACCAATCCCTCCAATCTTGGGAACTTTGTTCATGCTGCTCTACCCATGACCCGAACTTACCTTCTATTTTCCCTGCTACTGATCACTGTTTATTGAAATTCAGTTCATCCTTTAAAGCCCAGTTCAATCTTTCTTTCTTTCTTTTCTTTTTCTTTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTT 55+5555555555+55+5555+55+555555555+5555555555555555555555555+5555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555,5555555555555555555555555555 NM:i:0 MD:Z:250 AS:i:50000 XS:i:0"

def compute_sw(sam_file, reference_sequences):
    all_waterman_distances = []
    # Extract reference name and alignment start position
    with open(sam_file, "r") as f:
        lines = f.readlines()
    
    for sam_line in tqdm(lines):
        
        columns = sam_line.split('\t')
        if len(columns) < 5 or sam_line[0] == "@":
            continue
        
        ref_name = columns[2]
        alignment_start = int(columns[3])

        read_sequence = columns[9]
        reference_sequence = reference_sequences.get(ref_name, '')

        # Display aligned sequences side by side
        if reference_sequence:
            aligned_length = len(read_sequence)
            reference_fragment = reference_sequence[max(alignment_start-500,0): alignment_start + aligned_length + 500]
            try:
                all_waterman_distances.append(calculate_smith_waterman_distance(reference_fragment.upper(), read_sequence.upper())["distance"])
            except:
                print(aligned_length)
                print(read_sequence)
                print(reference_fragment)
                print(alignment_start)
        else:
            # print(sam_line)
            # print("Missed.")
            pass
        
        # if reference_sequence:
        #     print(read_sequence)
        #     print(reference_fragment)
        #     print(all_waterman_distances[-1])
        #     break
    print("SW distance:", np.mean(all_waterman_distances))
