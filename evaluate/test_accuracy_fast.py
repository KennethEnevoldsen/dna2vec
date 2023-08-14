from helpers import raw_fasta_files
import numpy as np
from tqdm import tqdm

from helpers import initialize_pinecone, is_within_range_of_any_element, main_align


from pathlib import Path
from datetime import datetime
import logging
import random
from itertools import product

from dna2vec.config_schema import DatasetConfigSchemaUniformSampling
from dna2vec.simulate import simulate_mapped_reads

from pinecone_store import PineconeStore

import os
os.environ["DNA2VEC_CACHE_DIR"] = "/mnt/SSD2/pholur/dna2vec"

grid = {
    "read_length": [500, 250], #[150, 300, 500],
    "insertion_rate": [0.0, 0.00009, 0.0009, 0.009],
    "deletion_rate" : [0.0, 0.00011, 0.0011, 0.011],
    "qq": [(60,80), (40,60), (20,40)],
    "topk": [5, 25, 50]
}


###
import argparse
parser = argparse.ArgumentParser(description="Grid Search")
parser.add_argument('--recipe', type=str)
parser.add_argument('--checkpoints', type=str)
parser.add_argument('--topk', type=int)
parser.add_argument('--test', type=int)
parser.add_argument('--system', type=str)
parser.add_argument('--exactness', type=int)
parser.add_argument('--distance_bound', type=int)
parser.add_argument('--device', type=str)
###




if __name__ == "__main__":
    
    args = parser.parse_args()
    fasta_file_path = raw_fasta_files[args.recipe]
    
    data_queue = args.recipe.split(";")
    checkpoint_queue = args.checkpoints.split(";")
    
    now = datetime.now()
    formatted_date = now.strftime("%Y_%m_%d_%H_%M_%S")

    logging.basicConfig(filename = Path(os.environ["DNA2VEC_CACHE_DIR"]) / "Logs" / f"log_{formatted_date}", 
                        level=logging.INFO)
    
    logging.info("Parameters:")
    
    f = open(Path(os.environ["DNA2VEC_CACHE_DIR"]) / "Results" / f"result_{formatted_date}.csv", "w+")
        
    for arg in vars(args):
        f.write(f"# {arg}: {getattr(args, arg)}\nqqq")
        
    f.write("Quality,Read length,Insertion rate,Deletion rate,TopK,Accuracy\n")
    f.flush()
    
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
        
    for (store, _, _) in initialize_pinecone(checkpoint_queue, data_queue, args.device):
        
        for read_length, insertion_rate, deletion_rate, quality, topk in \
            product(grid["read_length"], grid["insertion_rate"], grid["deletion_rate"], grid["qq"], grid["topk"]):

            perf_read = 0
            perf_true = 0
            count = 0

            queries = []
            small_indices = []
            start_indices = []
            
            mapped_reads = simulate_mapped_reads(
                n_reads_pr_amplicon=args.test,
                read_length=read_length,
                insertion_rate=insertion_rate,
                deletion_rate=deletion_rate,
                reference_genome=fasta_file_path,
                sequencing_system=args.system,
                quality = quality
            )
            
            for sample in tqdm(mapped_reads):
                queries.append(sample.read.query_sequence)
                small_indices.append(int(sample.read.reference_start_))
                start_indices.append(int(sample.seq_offset))
            
            ground_truth = [index_main + inter_fine for index_main, inter_fine in zip(start_indices, small_indices)]    
            results = main_align(store, queries, ground_truth, topk, exactness=args.exactness, distance_bound=args.distance_bound, flex = True)
            total_perf = np.mean(results)
            f.write(f"{str(quality).replace(',',';')},{read_length},{insertion_rate},{deletion_rate},{topk},{total_perf}\n")
            f.flush()
                        
    f.close()
                        
                    