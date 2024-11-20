import os
import argparse
import logging
import random
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
from tqdm import tqdm
import jsonlines

from helpers import configs, initialize_pinecone, is_within_range_of_any_element, main_align
from dna2vec.config_schema import DatasetConfigSchemaUniformSampling
from dna2vec.simulate import simulate_mapped_reads
from pinecone_store import PineconeStore

os.environ["DNA2VEC_CACHE_DIR"] = "/home/shreyas/NLP/dna2vec"

grid = {
    "read_length": [250], # Read length, ART has a max read length of 250
    "insertion_rate": [0.01], # Insertion rate, ART has a max insertion rate of 0.01
    "deletion_rate": [0.01], # Deletion rate, ART has a max deletion rate of 0.01
    "qq": [
        (30, 60),
        # (60, 90),
    ],  # https://www.illumina.com/documents/products/technotes/technote_Q-Scores.pdf
    "topk": [50], # Topk is the number of nearest neighbors to retrieve
    "distance_bound": [15],  # Room for SW distance, this is to compute correct_by_score
    "exactness": [2],  # Room for index, this is to compute correct_by_location
}


###
parser = argparse.ArgumentParser(description="Accuracy Testing for different parameter settings")
parser.add_argument("--vector_db", nargs="+", default=["all"], help="Specify the reference chromosome used to populate the vector db")
parser.add_argument("--model", nargs="+", default=["trained-all-reg-mean-pooling"], help="Specify the model used to generate the embeddings")
parser.add_argument("--num_reads", type=int, default=200, help="Specify the number of reads to test")
parser.add_argument("--system", type=str, default="MSv3", help="Specify the sequencing system to use for testing")
parser.add_argument("--device", type=str, default="cuda:0", help="Specify the device to use for testing")
parser.set_defaults(namespace=True)
parser.add_argument("--no-namespace", action="store_false", dest="namespace", help="Disable namespace (default: namespace is enabled)")
parser.add_argument("--reads", nargs="+", default=["all"], help="Specify which chromosome the reads are extracted from")
parser.add_argument("--compare_type", type=str, choices=["location", "score", "both"], default="both", help="Specify the type of comparison to use for testing")
parser.add_argument("--pod_type", type=str, default="s1.x1", help="Specify the pod type to use for testing, note this is important only if you are creating a new index")

###


def control_meta_map(path):
    """
    Read the meta data map from the given path.
    """
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

    read_references = args.reads

    for reference in read_references:

        fasta_file_path = configs["raw_fasta_files"][reference]

        vector_db_queue = args.vector_db
        model_queue = args.model

        namespace = args.namespace
        compare_type = args.compare_type
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

        f = open(res_folder / f"result_{formatted_date}.csv", "w+")

        # Write the datafile name
        f.write(f"# Datafile: {reference}\n")
        f.write(f"# read: {reference}\n")
        for arg in vars(args):
            if arg != 'reads':
                f.write(f"# {arg}: {getattr(args, arg)}\n")
        f.write(
            "Quality,Read length,Insertion rate,Deletion rate,TopK,Distance bound,Exactness,Accuracy,Error Lower Bound,Error Upper Bound\n"
        )
        f.flush()

        for arg in vars(args):
            logging.info(f"{arg}: {getattr(args, arg)}")

        print(model_queue, vector_db_queue, args.device, args.pod_type)
        for store, _, _ in initialize_pinecone(
            model_queue, vector_db_queue, args.device, args.pod_type
        ):
            for (
                read_length,
                insertion_rate,
                deletion_rate,
                quality,
                topk,
                distance_bound,
                exactness,
            ) in product(
                grid["read_length"],
                grid["insertion_rate"],
                grid["deletion_rate"],
                grid["qq"],
                grid["topk"],
                grid["distance_bound"],
                grid["exactness"],
            ):

                distributed = False
                per_k = 0

                if namespace:
                    print(
                    "WARNING: Enabling default equal sampling from all chromosomes"
                    )
                    print("WARNING: Currently only executes the hotstart.")
                    per_k = topk  # 25 chromosomes
                    topk = per_k * 25
                    distributed = True

                perf_read = 0
                perf_true = 0
                count = 0

                queries = []
                small_indices = []
                start_indices = []
                meta = []

                mapped_reads = simulate_mapped_reads(
                    n_reads_pr_amplicon=args.num_reads,
                    read_length=read_length,
                    insertion_rate=insertion_rate,
                    deletion_rate=deletion_rate,
                    reference_genome=fasta_file_path,
                    sequencing_system=args.system,
                    quality=quality,
                ) # ART generated reads

                print("MAPPED READS: ", len(mapped_reads))

                queries = []
                meta = []

                dictionary_of_values = {}
                for sample in tqdm(mapped_reads):
                    queries.append(sample.read.query_sequence)
                    small_indices.append(int(sample.read.reference_start))
                    start_indices.append(int(sample.seq_offset))
                    if namespace:
                        meta.append(meta_data_map[str(sample.id)])
                    dictionary_of_values[sample.read.query_sequence] = (
                        sample.reference.upper()
                    )

                ground_truth = [
                    index_main + inter_fine
                    for index_main, inter_fine in zip(start_indices, small_indices)
                ]
                if namespace:
                    print("NAMESPACE ENABLED")
                    print("META: ", len(meta))
                    results, lower_bound, upper_bound = main_align(
                        store,
                        queries,
                        ground_truth,
                        topk,
                        exactness=exactness,
                        distance_bound=distance_bound,
                        flex=True,
                        distributed=distributed,
                        per_k=per_k,
                        namespaces=meta,
                        namespace_dict=meta_data_map,
                        dictionary_of_values=dictionary_of_values,
                        compare_type=compare_type,
                    )

                else:
                    print("RUNNING WITHOUT NAMESPACE")
                    results, lower_bound, upper_bound = main_align(
                        store,
                        queries,
                        ground_truth,
                        topk,
                        exactness=exactness,
                        distance_bound=distance_bound,
                        flex=True,
                        distributed=distributed,
                        per_k=per_k,
                        namespaces=None,
                        namespace_dict=None,
                        dictionary_of_values=dictionary_of_values,
                        compare_type=compare_type,
                    )

                total_perf = np.mean(results)
                print("TOTAL PERF: ", total_perf)
                f.write(
                    f"{str(quality).replace(',',';')},{read_length},{insertion_rate},{deletion_rate},{topk},{distance_bound},{exactness},{total_perf},{lower_bound},{upper_bound}\n"
                )

                f.flush()
        f.close()
