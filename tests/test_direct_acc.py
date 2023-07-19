from dna2vec.simulate import simulate_mapped_reads
from helpers import raw_fasta_files
import numpy as np
from tqdm import tqdm
from helpers import bwamem_align, initialize_pinecone

grid = {
    "read_length": [150, 300, 500],
    "insertion_rate": [0.00009, 0.0009, 0.009],
    "deletion rate" : [0.00011, 0.0011, 0.011]
}

import argparse
parser = argparse.ArgumentParser(description="Grid Search")
parser.add_argument('--recipe', type=str)
parser.add_argument('--checkpoints', type=str)
parser.add_argument('--topk', type=int)
parser.add_argument('--test', type=int)

import os
os.environ["DNA2VEC_CACHE_DIR"] = "/mnt/SSD2/pholur/dna2vec"


import time
def evaluate(store, query, top_k):

        
    returned = store.query([query], top_k=top_k)["matches"]

    trained_positions = [sample["metadata"]["position"] for sample in returned]
    metadata_set = [sample["metadata"]["metadata"] for sample in returned]
    all_candidate_strings = [sample["metadata"]["text"] for sample in returned]
        
    identified_sub_indices, identified_indices, metadata_indices, timer = bwamem_align(
                                                                                all_candidate_strings, 
                                                                                trained_positions, 
                                                                                metadata_set,
                                                                                query)
    return [rough + fine for (rough, fine) in zip(identified_indices, identified_sub_indices)], timer



if __name__ == "__main__":
    
    args = parser.parse_args()
    fasta_file_path = raw_fasta_files[args.recipe]
    
    data_queue = args.recipe.split(";")
    checkpoint_queue = args.checkpoints.split(";")
    
    for (store, _, _) in initialize_pinecone(checkpoint_queue, data_queue):
        
        for read_length in grid["read_length"]:
            for insertion_rate in grid["insertion_rate"]:
                for deletion_rate in grid["deletion_rate"]:
                    
                    perf = 0
                    count = 0
                    
                    mapped_reads = simulate_mapped_reads(
                        n_reads_pr_amplicon=args.test,
                        read_length=read_length,
                        insertion_rate=insertion_rate,
                        deletion_rate=deletion_rate,
                        reference_genome=fasta_file_path
                    )
                    
                    for sample in mapped_reads:
                        
                        query = sample.read.query_sequence
                        beginning = sample.read.reference_start
                        matches, timer = evaluate(store, query, args.topk)
                        
                        if beginning in matches:
                            perf += 1
                        count += 1

                    print(read_length, insertion_rate, deletion_rate, perf/count)
                        
                    