import os

os.environ["DNA2VEC_CACHE_DIR"] = "/home/shreyas/NLP/dna2vec"

from helpers import raw_fasta_files
import numpy as np
from tqdm import tqdm

from helpers import (
    initialize_pinecone,
    is_within_range_of_any_element,
    main_align,
    align_real_reads,
)


from pathlib import Path
from datetime import datetime
import logging
import random
from itertools import product

from dna2vec.config_schema import DatasetConfigSchemaUniformSampling
from dna2vec.simulate import simulate_mapped_reads

from pinecone_store import PineconeStore
import jsonlines

# grid = {
#     "read_length": [250], #[150, 300, 500],
#     "insertion_rate": [0.0, 0.01],
#     "deletion_rate" : [0.0, 0.01],
#     "qq": [(60,90), (30,60)], # https://www.illumina.com/documents/products/technotes/technote_Q-Scores.pdf
#     "topk": [1250, 250], #[50, 100],
#     "distance_bound": [25, 5, 0],
#     "exactness": [2]
# }

# grid = {
#     "read_length": [250], #[150, 300, 500],
#     "insertion_rate": [0.01],
#     "deletion_rate" : [0.01],
#     "qq": [(30,60)], # https://www.illumina.com/documents/products/technotes/technote_Q-Scores.pdf
#     "topk": [250], #[50, 100],
#     "distance_bound": [50],
#     "exactness": [10]
# }


grid = {
    "topk": [50],  # [50, 100],
    "distance_bound": [0],
}


###
import argparse

parser = argparse.ArgumentParser(description="Grid Search")
parser.add_argument("--recipe", type=str)
parser.add_argument("--checkpoints", type=str)
parser.add_argument("--device", type=str)
parser.add_argument("--namespace", type=bool)
###

PATH_TO_REAL_READS = "/mnt/SSD1/shreyas/dna2vec/data/sample_250_reads.txt"
# PATH_TO_REAL_READS = "test_cache/queries_lowphread.txt"
# PATH_TO_NAMESPACES = "test_cache/meta_lowphread.txt"
GLOBAL_READ_NAMESPACE_ID = "chr2"


# Mapping for namespaces
def control_meta_map(path):
    meta_dict = {}
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        key = line.split(">")[1].split(" ")[0]
        meta_dict[key] = line.strip()

    return meta_dict


if __name__ == "__main__":

    meta_data_map = control_meta_map("test_cache/logs/headers")

    args = parser.parse_args()

    data_queue = args.recipe.split(";")
    checkpoint_queue = args.checkpoints.split(";")
    namespace = False

    now = datetime.now()
    formatted_date = now.strftime("%Y_%m_%d_%H_%M_%S")

    log_folder = Path(os.environ["DNA2VEC_CACHE_DIR"]) / "Logs"

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    logging.basicConfig(
        filename=log_folder / f"log_{formatted_date}", level=logging.INFO
    )

    logging.info("Parameters:")

    res_folder = Path(os.environ["DNA2VEC_CACHE_DIR"]) / "Results"

    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    f = open(res_folder / f"Pure_Reads_{formatted_date}.csv", "w+")

    for arg in vars(args):
        f.write(f"# {arg}: {getattr(args, arg)}\n")

    f.write("Writing Real Reads\n")
    f.write("Distance bound,Exactness,Accuracy\n")
    f.flush()

    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    for store, _, _ in initialize_pinecone(checkpoint_queue, data_queue, args.device):
        i = 0
        for (
            topk,
            distance_bound,
        ) in product(
            grid["topk"],
            grid["distance_bound"],
        ):

            # for chr_meta in meta_data_map.keys():

            # print("Checking for ", chr_meta)
            perf_read = 0
            perf_true = 0
            count = 0

            queries = []
            meta = []
            # read queries from file and store as list
            with open(PATH_TO_REAL_READS, "r") as read_file:
                for line in read_file:
                    line = line.strip()
                    line = line.upper()
                    if len(line) > 200:
                        queries.append(line)
                    if namespace:
                        meta.append(meta_data_map[GLOBAL_READ_NAMESPACE_ID])

            queries = queries[:200]
            # if namespace:
            #     with open(PATH_TO_NAMESPACES, "r") as read_file:
            #         for line in read_file:
            #             line = line.strip()
            #             meta.append(meta_data_map[line])

            results = align_real_reads(store, queries, topk, distance_bound, meta=meta)

            total_perf = np.mean(results)
            old_sum = np.sum(results)
            old_len = len(results)

            print("Total Perf: ", total_perf)
            # print(f"{str(quality).replace(',',';')},{read_length},{insertion_rate},{deletion_rate},{topk},{distance_bound},{exactness},{total_perf}\n")
            # exit()
            f.write(f"{topk},{distance_bound},{total_perf}\n")

            f.flush()

    f.close()
