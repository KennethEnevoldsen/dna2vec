from helpers import raw_fasta_files
import numpy as np
from tqdm import tqdm
from helpers import bwamem_align, bwamem_align_parallel, initialize_pinecone
from pathlib import Path
from datetime import datetime
import logging
import random

from dna2vec.config_schema import DatasetConfigSchemaUniformSampling
from dna2vec.simulate import simulate_mapped_reads

from pinecone_store import PineconeStore

import os
os.environ["DNA2VEC_CACHE_DIR"] = "/mnt/SSD2/pholur/dna2vec"

grid = {
    "read_length": [250], #[150, 300, 500],
    "insertion_rate": [0.00009, 0.0009, 0.009],
    "deletion_rate" : [0.00011, 0.0011, 0.011],
    "qq": [(20,40), (40,60), (60,80)]
}


###
import argparse
parser = argparse.ArgumentParser(description="Grid Search")
parser.add_argument('--recipe', type=str)
parser.add_argument('--checkpoints', type=str)
parser.add_argument('--topk', type=int)
parser.add_argument('--test', type=int)
parser.add_argument('--system', type=str)
###




def evaluate(store: PineconeStore, 
             query: str, 
             top_k: int
):
    returned = store.query([query], top_k=top_k)["matches"]
    trained_positions = [sample["metadata"]["position"] for sample in returned]
    metadata_set = [sample["metadata"]["metadata"] for sample in returned]
    all_candidate_strings = [sample["metadata"]["text"] for sample in returned]
    
    # Parallelized BWA mem - FAST
    identified_sub_indices, identified_indices, _, timer = bwamem_align_parallel(
                                                                                all_candidate_strings, 
                                                                                trained_positions, 
                                                                                metadata_set,
                                                                                query)

    return [int(rough) + int(fine) 
            for (rough, fine) in zip(identified_indices, identified_sub_indices)], timer


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
        f.write(f"# {arg}: {getattr(args, arg)}\n")
        
    f.write("Quality,Read length,Insertion rate,Deletion rate,Accuracy\n")
    f.flush()
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
        
    for (store, _, _) in initialize_pinecone(checkpoint_queue, data_queue):
        
        for read_length in grid["read_length"]:
            for insertion_rate in grid["insertion_rate"]:
                for deletion_rate in grid["deletion_rate"]:
                    for quality in grid["qq"]:
                        
                        perf_read = 0
                        perf_true = 0
                        count = 0
                        
                        mapped_reads = simulate_mapped_reads(
                            n_reads_pr_amplicon=args.test,
                            read_length=read_length,
                            insertion_rate=insertion_rate,
                            deletion_rate=deletion_rate,
                            reference_genome=fasta_file_path,
                            sequencing_system=args.system,
                            quality = quality
                        )
                        
                        random.shuffle(mapped_reads)
                        
                        for sample in tqdm(mapped_reads):
                            
                            query = sample.read.query_sequence
                            beginning = sample.read.reference_start
                            matches, timer = evaluate(store, query, args.topk)
                            if beginning in matches:
                                perf_read += 1
                            
                            # query = sample.reference
                            # beginning = sample.read.reference_start
                            # matches, timer = evaluate(store, query, args.topk)
                            # if beginning in matches:
                            #     perf_true += 1
      
                            count += 1
                        
                        
                        f.write(f"{str(quality).replace(',',';')},{read_length},{insertion_rate},{deletion_rate},{perf_read/count}\n")
                        f.flush()
                        # logging.info("\n\n ###############################################")
                        # logging.info(f"Read length: {read_length}, \
                        #             Insertion rate: {insertion_rate}, \
                        #             Deletion rate: {deletion_rate}, \
                        #             Accuracy: read - {perf_read/count}, true - {perf_true/count}")
                        # logging.info("###############################################\n\n")
                        
    f.close()
                        
                    