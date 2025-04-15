import hydra
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import product
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from helpers import initialize_pinecone, align_real_reads, query_and_align, post_process_results, sv_results_refiner
from dna2vec.simulate import real_mapped_reads #simulate_mapped_reads

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

    # Load metadata headers
    meta_data_map = load_meta_headers(cfg.paths.meta_headers)

    # Set up logging to use Hydra's output directory
    log_folder = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)  # Hydra changes working directory to the output folder
    experiment_specs_folder = log_folder / "experiment_specs"
    experiment_specs_folder.mkdir(parents=True, exist_ok=True)
    
    # Save configuration to experiment_specs folder
    with open(experiment_specs_folder / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    configs = cfg.configs

    for _ in cfg.reads:
        fasta_file_path = configs["raw_fasta_files"]
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
                    bam_file = Path(cfg.paths.bam_file)
                    mapped_reads = real_mapped_reads(
                        bam_file=bam_file,
                        reference_genome=fasta_file_path,
                        chr_number=cfg.experiment_settings.chr_number,
                        type_of_sv=cfg.experiment_settings.type_of_sv,
                        size_of_sv=cfg.experiment_settings.size_of_sv,
                        max_reads=cfg.experiment_settings.max_reads,
                        vcf_path=f"{cfg.base_settings.data_path}/human_deletions_GS_correct.vcf",
                    )

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
                    
                    if cfg.namespace:
                        processed_results, results_df = post_process_results(results_list, mapped_reads, queries, topk / 25)
                    else:
                        processed_results, results_df = post_process_results(results_list, mapped_reads, queries, topk)
                    
                    # Save the dataframe to CSV - using Hydra's output directory
                    results_dir = log_folder / f"results_chr{cfg.experiment_settings.chr_number}_{cfg.experiment_settings.type_of_sv}_{cfg.experiment_settings.size_of_sv}_{cfg.experiment_settings.max_reads}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
                    results_dir.mkdir(parents=True, exist_ok=True)
                    results_df_file = results_dir / f"results_unformatted.csv"
                    results_df.to_csv(results_df_file, index=False)
                    print(f"Results dataframe saved to {results_df_file}")
                    
                    results_df = sv_results_refiner(results_df_file)
                    refined_results_df_file = results_dir / f"results_refined.csv"
                    results_df.to_csv(refined_results_df_file, index=False)
                    print(f"Refined results dataframe saved to {refined_results_df_file}")
            
                    
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