import os
import hydra
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import product
from omegaconf import DictConfig
from tqdm import tqdm

from helpers import initialize_pinecone, align_real_reads, query_and_align
from dna2vec.simulate import simulate_mapped_reads


# Ensure environment variables are set for DNA2VEC
def set_env(dna2vec_cache_dir):
    os.environ["DNA2VEC_CACHE_DIR"] = dna2vec_cache_dir


# Read metadata headers for namespace alignment
def load_meta_headers(path):
    meta_dict = {}
    with open(path, "r") as f:
        for line in f.readlines():
            key = line.split(">")[1].split(" ")[0]
            meta_dict[key] = line.strip()
    return meta_dict


@hydra.main(config_path="configs", config_name="refined_evaluate", version_base=None)
def main(cfg: DictConfig):
    """Hydra-based experiment runner"""
    PATH_TO_REAL_READS = cfg.path_to_real_reads
    GLOBAL_READ_NAMESPACE_ID = cfg.global_read_namespace_id

    # Set up environment variables
    set_env(cfg.paths.dna2vec_cache_dir)

    # Load metadata headers
    meta_data_map = load_meta_headers(cfg.paths.meta_headers)

    # Create timestamp for logging and results
    formatted_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Set up logging
    log_folder = Path(cfg.paths.log_dir)
    log_folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_folder / f"log_{formatted_date}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Experiment Parameters:")
    for key, value in cfg.items():
        logging.info(f"{key}: {value}")

    # Create results directory
    res_folder = Path(cfg.paths.results_dir)
    res_folder.mkdir(parents=True, exist_ok=True)
    
    configs = cfg.configs

    for reference in cfg.reads:
        fasta_file_path = configs["raw_fasta_files"][reference]

        results_file = res_folder / f"result_{formatted_date}.csv"
        with open(results_file, "w") as f:
            f.write(f"# Datafile: {reference}\n")
            f.write(f"# Read: {reference}\n")
            for key, value in cfg.items():
                if key != "reads":
                    f.write(f"# {key}: {value}\n")
            f.write(
                "Quality,Read length,Insertion rate,Deletion rate,TopK,Distance bound,Exactness,Accuracy,Error Lower Bound,Error Upper Bound\n"
            )

            # Initialize Pinecone storage
            for store, _, _ in initialize_pinecone(cfg.model, cfg.vector_db, cfg.device, cfg.pod_type):
                for (
                    read_length,
                    insertion_rate,
                    deletion_rate,
                    quality,
                    topk,
                    distance_bound,
                    exactness,
                ) in product(
                    cfg.grid_search.read_length,
                    cfg.grid_search.insertion_rate,
                    cfg.grid_search.deletion_rate,
                    cfg.grid_search.qq,
                    cfg.grid_search.topk,
                    cfg.grid_search.distance_bound,
                    cfg.grid_search.exactness,
                ):

                    distributed = False
                    per_k = 0

                    if cfg.namespace:
                        print("WARNING: Enabling equal sampling from all chromosomes (hotstart mode)")
                        per_k = topk
                        topk = per_k * 25
                        distributed = True
                    if cfg.test_mode:
                        # Generate ART reads (simulated)
                        mapped_reads = simulate_mapped_reads(
                            n_reads_pr_amplicon=cfg.num_reads,
                            read_length=read_length,
                            insertion_rate=insertion_rate,
                            deletion_rate=deletion_rate,
                            reference_genome=fasta_file_path,
                            sequencing_system=cfg.system,
                            quality=quality,
                        )
                        print(f"MAPPED READS: {len(mapped_reads)}")

                        queries = []
                        small_indices = []
                        start_indices = []
                        meta = []
                        dictionary_of_values = {}

                        for sample in tqdm(mapped_reads):
                            queries.append(sample.read.query_sequence)
                            small_indices.append(int(sample.read.reference_start))
                            start_indices.append(int(sample.seq_offset))
                            if cfg.namespace:
                                meta.append(meta_data_map[str(sample.id)])
                            dictionary_of_values[sample.read.query_sequence] = sample.reference.upper()

                        ground_truth = [
                            index_main + inter_fine
                            for index_main, inter_fine in zip(start_indices, small_indices)
                        ]

                        print("Running alignment...")
                        results, lower_bound, upper_bound = query_and_align(
                            store,
                            queries,
                            ground_truth,
                            topk,
                            exactness=exactness,
                            distance_bound=distance_bound,
                            flex=True,
                            distributed=distributed,
                            per_k=per_k,
                            namespaces=meta if cfg.namespace else None,
                            namespace_dict=meta_data_map if cfg.namespace else None,
                            dictionary_of_values=dictionary_of_values,
                            compare_type=cfg.compare_type,
                            return_type=cfg.return_type,
                        )
                        
                        if cfg.return_type == "score":
                            total_perf = np.mean(results)
                            print(f"TOTAL PERFORMANCE: {total_perf:.4f}")
                            print(f"LOWER BOUND: {lower_bound:.4f}")
                            print(f"UPPER BOUND: {upper_bound:.4f}")
                            
                            return results, total_perf, lower_bound, upper_bound

                        else:
                            alignments = results
                            return alignments
                        

                        # # Write results
                        # f.write(
                        #     f"{str(quality).replace(',',';')},{read_length},{insertion_rate},{deletion_rate},{topk},{distance_bound},{exactness},{total_perf},{lower_bound},{upper_bound}\n"
                        # )
                        # f.flush()
                        
                    else:
                        # TODO: Add real reads reading here
                        namespace = cfg.namespace
                        queries = []
                        meta = []
                        with open(PATH_TO_REAL_READS, "r") as read_file:
                            for line in read_file:
                                line = line.strip()
                                line = line.upper()
                                if len(line) > 200:
                                    queries.append(line)
                                if namespace:
                                    meta.append(meta_data_map[GLOBAL_READ_NAMESPACE_ID])
                                    
                        results = align_real_reads(store, queries, topk, distance_bound, meta=meta)
                        total_perf = np.mean(results)

                        print("Total Perf: ", total_perf)
                        f.write(f"{topk},{distance_bound},{total_perf}\n")

                        f.flush()


if __name__ == "__main__":
    main()
