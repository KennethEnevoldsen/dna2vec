import os
import hydra
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product
from omegaconf import DictConfig
from tqdm import tqdm
from helpers import initialize_pinecone, align_real_reads, query_and_align, calculate_SW_and_Cosine_similarity
from dna2vec.simulate import real_mapped_reads #simulate_mapped_reads
import matplotlib.pyplot as plt

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
                        # mapped_reads = simulate_mapped_reads(
                        #     n_reads_pr_amplicon=cfg.num_reads,
                        #     read_length=read_length,
                        #     insertion_rate=insertion_rate,
                        #     deletion_rate=deletion_rate,
                        #     reference_genome=fasta_file_path,
                        #     sequencing_system=cfg.system,
                        #     quality=quality,
                        # )
                        # bam_file = Path("/mnt/SSD1/yigit/dna_data/rde2/HG002.GRCh38.2x250.bam")
                        bam_file = Path("/mnt/SSD1/yigit/dna_data/rde2/HG002.hs37d5.2x250.bam")
                        mapped_reads = real_mapped_reads(
                            bam_file=bam_file,
                            reference_genome=fasta_file_path,
                        )
                        
                        print(f"MAPPED READS: {len(mapped_reads)}")
                        
                        distance_df = pd.DataFrame(columns=["Read", "Reference", "Smith-Waterman Similarity"])
                        for read in mapped_reads:
                            sw_score, cosine_score = calculate_SW_and_Cosine_similarity(read.read.query_sequence, read.reference.upper())
                            match_start_index = read.read.reference_start
                            distance_df = pd.concat([distance_df, pd.DataFrame([{"Read": read.read.query_sequence,"Start Index":match_start_index, "Smith-Waterman Similarity": sw_score, "CIGAR String": read.read.cigarstring, "CIGAR Tuples": read.read.cigartuples}])], ignore_index=True)
                        
                        print(f"Number of items: {distance_df.shape[0]}")
                        
                        print(f"Number of items which has SW distance < -495: {distance_df[distance_df['Smith-Waterman Similarity'] < -495].shape[0]}")
                        print(f"Number of items which has SW distance < -400: {distance_df[distance_df['Smith-Waterman Similarity'] < -400].shape[0]}")
                        print(f"Number of items which has SW distance < -300: {distance_df[distance_df['Smith-Waterman Similarity'] < -300].shape[0]}")
                        print(f"Number of items which has SW distance < -200: {distance_df[distance_df['Smith-Waterman Similarity'] < -200].shape[0]}")
                        print(f"Number of items which has SW distance < -100: {distance_df[distance_df['Smith-Waterman Similarity'] < -100].shape[0]}")
                        print(f"Number of items which has SW distance < -50: {distance_df[distance_df['Smith-Waterman Similarity'] < -50].shape[0]}")
                        
                        # we need the indexes of the items which has SW distance < -495
                        indexes = distance_df[distance_df['Smith-Waterman Similarity'] < -495].index
                        print(f"Indexes of items which has SW distance < -495: {indexes}")
                        
                        refined_mapped_reads = [mapped_reads[i] for i in indexes]
                        their_distance_df = distance_df #.iloc[indexes]
                        
                        # Plot Smith-Waterman similarity as bins for -500 to -400 in 100 bins
                        plt.hist(distance_df['Smith-Waterman Similarity'], bins=100, range=(-500, -400))
                        plt.title("Smith-Waterman similarity")
                        plt.xlabel("Smith-Waterman similarity")
                        plt.ylabel("Number of items")
                        plt.savefig(res_folder / f"smith_waterman_similarity_{formatted_date}.png")
                        plt.close()
                        
                        plt.hist(distance_df['Smith-Waterman Similarity'], bins=100, range=(-500, -450))
                        plt.title("Smith-Waterman similarity")
                        plt.xlabel("Smith-Waterman similarity")
                        plt.ylabel("Number of items")
                        plt.savefig(res_folder / f"smith_waterman_similarity_450_to_500_{formatted_date}.png")
                        plt.close()
                        
                        # Plot Cosine similarity as bins for 0.8 to 1 in 100 bins
                        # plt.hist(distance_df['Cosine Similarity'], bins=100, range=(0.8, 1))
                        # plt.title("Cosine similarity")
                        # plt.xlabel("Cosine similarity")
                        # plt.ylabel("Number of items")
                        # plt.savefig(res_folder / f"cosine_similarity_{formatted_date}.png")
                        # plt.close()
                        
                        distance_df.to_csv(res_folder / f"distance_df_{formatted_date}.csv", index=False)

                        queries = []
                        small_indices = []
                        start_indices = []
                        meta = []
                        dictionary_of_values = {}

                        for sample in tqdm(mapped_reads): # TODO: Change back to mapped_reads
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
                        results, lower_bound, upper_bound, our_distance_df = query_and_align(
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
                        
                        
                        #Save our distance df and their distance df
                        our_distance_df.to_csv(res_folder / f"alignment_results_{formatted_date}.csv", index=False)
                        their_distance_df.to_csv(res_folder / f"ground_truth_{formatted_date}.csv", index=False)
                        
                        # merge our distance df and their distance df based on the read column and column name should be their_sw_distance
                        merged_distance_df = pd.merge(our_distance_df, their_distance_df, on="Read", suffixes=("_alignment_results", "_ground_truth"))
                        # add column is_RDE_better which is 1 if our_sw_distance < their_sw_distance and 0 otherwise
                        merged_distance_df["is_RDE_better"] = (merged_distance_df["Smith-Waterman Similarity_alignment_results"] <= merged_distance_df["Smith-Waterman Similarity_ground_truth"]).astype(int)
                        # reorder the columns so that the columns are: Read, their_Reference, our_Reference, their_sw_distance, our_sw_distance, their_cosine_similarity, our_cosine_similarity, is_RDE_better
                        print(merged_distance_df.columns)
                        merged_distance_df = merged_distance_df[["Read","Start Index_ground_truth", "Start Index_alignment_results", "Smith-Waterman Similarity_ground_truth", "Smith-Waterman Similarity_alignment_results", "is_RDE_better","CIGAR Tuples", "CIGAR String"]]
                        merged_distance_df.to_csv(res_folder / f"merged_distance_df_{formatted_date}.csv", index=False)
                        
                        if cfg.return_type == "score":
                            total_perf = np.mean(results)
                            print(f"TOTAL PERFORMANCE: {total_perf:.4f}")
                            print(f"LOWER BOUND: {lower_bound:.4f}")
                            print(f"UPPER BOUND: {upper_bound:.4f}")
                            
                            # print(f"TOTAL PERFORMANCE REAL: {total_perf_real:.4f}")
                            
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
