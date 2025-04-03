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
from helpers import initialize_pinecone, align_real_reads, query_and_align
from dna2vec.simulate import real_mapped_reads #simulate_mapped_reads

# Post-process results to check if match.start is within [index, index+1250]
def post_process_results(results, mapped_reads, queries):
    """
    For each dictionary in results, check if the corresponding match.read.reference_start 
    is within [index, index+1250] for any index in the dictionary's indices field.
    
    Args:
        results: List of dictionaries from query_and_align
        mapped_reads: List of ReadAndReference objects
        queries: List of query sequences
        
    Returns:
        DataFrame with columns for read, result, reference_interval, and cigar_string
        List of 1s and 0s indicating if the condition is met for each query
    """
    processed_results = []
    df_data = []
    
    # Create a mapping from query sequence to mapped_read
    query_to_read = {read.read.query_sequence: read for read in mapped_reads}
    
    for i, result_dict in enumerate(results):
        query = result_dict["query"]
        all_candidate_strings = result_dict["fragments"]
        index_to_trained_positions = result_dict["trained_positions"]
        indices = result_dict["indices"]
        
        # Get the corresponding mapped_read
        mapped_read = query_to_read.get(query)
        
        if mapped_read and index_to_trained_positions:
            # Check if reference_start is within [index, index+1250] for any index
            read_reference_start = mapped_read.read.reference_start
            cigar_string = mapped_read.read.cigarstring
            
            # Find the matching index (if any)
            matching_trained_position = None
            best_distance = None
            best_fragment_distance = None
            best_fragment = None
            is_top_match = False
            topk_index = None
            is_best_frag_dist_aligns_best_sw_dist = False
            best_fragment_start_index = None
            is_best_frag_start_idx_eq_read_ref_start_idx = False
            # sort trained_positions by index which is the key of the dictionary
            index_to_trained_positions = {index: index_to_trained_positions[index] for index in indices}
            returned_topk_data = [data for key, data in dict(result_dict["distance_to_index"]).items()][:1000]
            returned_topk_data = [data_value[0] for data_value in returned_topk_data]
            # sort returned_topk_data wrt all_candidate_strings, every data_value is a tuple (index,index, _, candidate_string)
            returned_topk_data_sorted = sorted(returned_topk_data, key=lambda x: all_candidate_strings.index(x[3]))
            for enum, (index_start,trained_pos,_,candidate_string,_) in enumerate(returned_topk_data_sorted):
                index = index_start + int(trained_pos)
                trained_position = index_to_trained_positions[index]
                if trained_position <= read_reference_start <= trained_position + 1250:
                    matching_trained_position = trained_position
                    is_top_match = True if enum == 0 else False
                    # find the index of the index in the sorted index_to_distance wrt distance
                    topk_index = enum
                    best_fragment = candidate_string
                    best_distance = min(result_dict["distances"])
                    best_fragment_distance = result_dict["index_to_distance"][index][0]
                    is_best_frag_dist_aligns_best_sw_dist = True if best_fragment_distance == best_distance else False
                    best_fragment_start_index = index
                    is_best_frag_start_idx_eq_read_ref_start_idx = True if best_fragment_start_index == read_reference_start else False
            # Set result to 1 if there's a matching index, 0 otherwise
            result = 1 if matching_trained_position is not None else 0
            processed_results.append(result)
            
            # Determine reference interval
            reference_interval = f"[{matching_trained_position}, {matching_trained_position + 1250}]" if matching_trained_position is not None else "No match"
            
            # Add row to dataframe data
            df_data.append({
                "read": query,  # Read sequence
                "read_reference_start_index": read_reference_start,  # Index where the read starts in the reference
                "cigar_string": cigar_string,  # CIGAR string
                "reference_interval": reference_interval,  # Reference interval
                "is_exists_in_topk_fragments": result,  # Is the read in the topk fragments
                
                # TopK match information
                "topk_index": topk_index,  # Index of the topk match
                "is_top_match": is_top_match,  # Is top match in topk
                
                # Fragment information
                "best_fragment": best_fragment,  # Best fragment w.r.t position
                "best_fragment_start_index": best_fragment_start_index,  # Index of the best fragment
                
                # Distance metrics
                "best_distance_in_topk": best_distance,  # Best distance in topk
                "best_fragment_distance": best_fragment_distance,  # Distance of the best fragment w.r.t position
                "is_best_frag_dist_aligns_best_sw_dist": is_best_frag_dist_aligns_best_sw_dist,  # Is the best fragment distance aligns with the best sw distance
                "is_best_fragment_start_index_equal_to_index": is_best_frag_start_idx_eq_read_ref_start_idx  # Is the best fragment start index equals to the read reference start index
            })  
        else:
            processed_results.append(0)
            # Add row to dataframe data for reads with no matches
            df_data.append({
                "read": query,
                "result": 0,
                "reference_start": "N/A",
                "reference_interval": "No match",
                "cigar_string": "N/A" if not mapped_read else mapped_read.read.cigarstring
            })
    
    # Create dataframe
    results_df = pd.DataFrame(df_data)
    
    return processed_results, results_df

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
                        results, lower_bound, upper_bound, results_list = query_and_align(
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
                        
                        processed_results, results_df = post_process_results(results_list, mapped_reads, queries)
                        
                        # Save the dataframe to CSV
                        results_dir = Path(cfg.paths.results_dir)
                        results_dir.mkdir(parents=True, exist_ok=True)
                        results_df_file = results_dir / f"results_df_{formatted_date}_topk_{topk}.csv"
                        results_df.to_csv(results_df_file, index=False)
                        print(f"Results dataframe saved to {results_df_file}")
                
                        
                        if cfg.return_type == "score":
                            total_perf = np.mean(results)
                            print(f"TOTAL PERFORMANCE: {total_perf:.4f}")
                            print(f"LOWER BOUND: {lower_bound:.4f}")
                            print(f"UPPER BOUND: {upper_bound:.4f}")
                            
                            # Additional metrics from post-processing
                            total_post_perf = np.mean(processed_results)
                            print(f"POST-PROCESSING PERFORMANCE: {total_post_perf:.4f}")
                            
                            return results, total_perf, lower_bound, upper_bound, results_df

                        else:
                            alignments = results
                            # Calculate accuracy from post-processed results
                            total_post_perf = np.mean(processed_results)
                            print(f"POST-PROCESSING PERFORMANCE: {total_post_perf:.4f}")
                            return alignments, results_df
                        

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
