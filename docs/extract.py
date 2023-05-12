"""
minimal script for processing human genome to txt files
"""

# !pip install biopython
from Bio import SeqIO

# Open the FASTA file and parse the sequences
fasta_file = "/Users/au561649/Github/dna2vec/data/GRCh38_latest_genomic.fna"
sequences = SeqIO.parse(open(fasta_file), "fasta")

# Loop over the sequences and extract the one for chromosome 18
for seq_record in sequences:
    # for each ID write it to the output file for a total of 30 files
    id_ = seq_record.id
    seq = seq_record.seq
    print(id_)
    if "NC_000002.12" == id_:
        with open(f"/Users/au561649/Github/dna2vec/data/{id_}.txt", "w") as f:
            f.write(str(seq))
