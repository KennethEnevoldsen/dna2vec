# Standard library imports
import random
from typing import List, Tuple
# Third-party imports
import jsonlines
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm
import scipy.stats as stats
import pandas as pd
import os
# Local imports
from aligners.smith_waterman import bwamem_align_parallel, calculate_smith_waterman_distance
from dna2vec.model import model_from_config
from inference_models import EvalModel
from pinecone_store import PineconeStore
from dna2vec.config_schema import ModelConfigSchema
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn


# Define the configuration files
file_dir = os.path.dirname(os.path.abspath(__file__))
config_files = {
    "data_recipes": f"{file_dir}/configs/data_recipes.yaml",
    "checkpoints": f"{file_dir}/configs/model_checkpoints.yaml",
    "raw_fasta_files": f"{file_dir}/configs/raw.yaml"
}

def load_yaml_config(file_path):
    """
    Load a YAML configuration file.
    """
    with open(file_path, "r") as stream:
        return yaml.safe_load(stream)

# Load all configurations
configs = {key: load_yaml_config(path) for key, path in config_files.items()}

# Now you can access the configurations like this:
# data_recipes = configs["data_recipes"]
# checkpoints = configs["checkpoints"]
# raw_fasta_files = configs["raw_fasta_files"]

def load_hf_model():
    hf_model = AutoModel.from_pretrained("roychowdhuryresearch/dna2vec", trust_remote_code=True)
    hf_tokenizer = AutoTokenizer.from_pretrained("roychowdhuryresearch/dna2vec", trust_remote_code=True)

    class AveragePooler(nn.Module):
        """
        Parameter-free poolers to get the sentence embedding
        # derived from https://github.com/princeton-nlp/SimCSE/blob/13361d0e29da1691e313a94f003e2ed1cfa97fef/simcse/models.py#LL49C1-L84C1
        """

        def __init__(self):
            super().__init__()

        def forward(self, last_hidden, attention_mask):
            # Old previous implementation
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
                -1
            ).unsqueeze(-1)

    hf_model.pooler = AveragePooler()
    return hf_model, hf_tokenizer, hf_model.pooler

def generate_cigar_string(aligned_indices, query_length=None):
    """
    Generate CIGAR string from alignment indices returned by BioPython's PairwiseAlign 
    with Smith-Waterman algorithm, including soft clipping.
    
    Args:
        aligned_indices: A numpy array containing two arrays of start/end indices:
                        [[(t_start1, t_end1), ...], [(q_start1, q_end1), ...]]
                        where t refers to target and q refers to query sequence indices.
        query_length: Optional; the total length of the query sequence for soft clipping.
                     If not provided, soft clips will only be added if clearly needed.
    
    Returns:
        A CIGAR string representing the alignment (e.g., "3S5M2I3M1D7M4S")
    """
    if not isinstance(aligned_indices, np.ndarray) or len(aligned_indices) != 2:
        print("Warning: Input must be an array of two arrays.")
        return "*"
    
    target_indices, query_indices = aligned_indices
    
    if len(target_indices) != len(query_indices):
        raise ValueError("Target and query must have the same number of aligned segments")
    
    if len(target_indices) == 0:
        return "*"  # Return unaligned marker if no alignments
    
    cigar = []
    
    # Add soft clip at the beginning if the alignment doesn't start at the beginning of the query
    if query_indices[0][0] > 0:
        cigar.append(f"{query_indices[0][0]}S")
    
    last_t_end = None
    last_q_end = None
    
    for i in range(len(target_indices)):
        t_start, t_end = target_indices[i]
        q_start, q_end = query_indices[i]
        
        # Handle gaps between aligned segments
        if last_t_end is not None and last_q_end is not None:
            t_gap = t_start - last_t_end
            q_gap = q_start - last_q_end
            
            # Add deletion (gap in query)
            if t_gap > q_gap:
                del_size = t_gap - q_gap
                cigar.append(f"{del_size}D")
            
            # Add insertion (gap in target)
            elif q_gap > t_gap:
                ins_size = q_gap - t_gap
                cigar.append(f"{ins_size}I")
        
        # Add current match
        match_length = min(t_end - t_start, q_end - q_start)
        if match_length > 0:
            cigar.append(f"{match_length}M")
        
        last_t_end = t_end
        last_q_end = q_end
    
    # Add soft clip at the end if the alignment doesn't end at the end of the query
    if query_length is not None and query_indices[-1][1] < query_length:
        soft_clip_end = query_length - query_indices[-1][1]
        if soft_clip_end > 0:
            cigar.append(f"{soft_clip_end}S")
    
    return "".join(cigar)

# Post-process results to check if match.start is within [index, index+1250]
def post_process_results(results, mapped_reads, queries, topk):
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
            best_alignment_str = None
            is_best_frag_start_idx_eq_read_ref_start_idx = False
            # sort trained_positions by index which is the key of the dictionary
            index_to_trained_positions = {index: index_to_trained_positions[index] for index in indices}
            returned_topk_data = [data for key, data in dict(result_dict["distance_to_index"]).items()][:int(topk)] 
            returned_topk_data = [data_value[0] for data_value in returned_topk_data]
            # sort returned_topk_data wrt all_candidate_strings, every data_value is a tuple (index,index, _, candidate_string)
            returned_topk_data_sorted = sorted(returned_topk_data, key=lambda x: all_candidate_strings.index(x[3]))
            candidate_rows = []
            for enum, (index_start,trained_pos,_,candidate_string,_,alignment_str, alignment_indices) in enumerate(returned_topk_data_sorted):
                index = index_start + int(trained_pos)
                if int(trained_pos) <= read_reference_start <= int(trained_pos) + 1300:
                    matching_trained_position = int(trained_pos)
                    is_top_match = True if enum == 0 else False
                    # find the index of the index in the sorted index_to_distance wrt distance
                    topk_index = enum
                    best_fragment = candidate_string
                    best_distance = min(result_dict["distances"])
                    sorted_best_three_distances = sorted(result_dict["distances"])[:3]
                    sorted_best_three_alignment_str = [result_dict["distance_to_index"][distance][0][-2] for distance in sorted_best_three_distances]
                    sorted_best_three_alignment_indices = [result_dict["distance_to_index"][distance][0][-1] for distance in sorted_best_three_distances]
                    sorted_best_three_alignment_cigar_strings = [generate_cigar_string(alignment_indices) for alignment_indices in sorted_best_three_alignment_indices]
                    sorted_best_three_alignment_start_indices = [result_dict["distance_to_index"][distance][0][0] + int(result_dict["distance_to_index"][distance][0][1]) for distance in sorted_best_three_distances]
                    sorted_alignment_integrity = [calculate_alignment_integrity(alignment_indices) for alignment_indices in sorted_best_three_alignment_indices]
                    best_fragment_distance = result_dict["index_to_distance"][index][0]
                    best_fragment_cigar_string = generate_cigar_string(result_dict["distance_to_index"][best_fragment_distance][0][-1])
                    best_alignment_str = alignment_str
                    is_best_frag_dist_aligns_best_sw_dist = True if best_fragment_distance == best_distance else False

                    best_fragment_start_index = index
                    is_best_frag_start_idx_eq_read_ref_start_idx = True if best_fragment_start_index == read_reference_start else False
                    reference_interval = f"[{matching_trained_position}, {matching_trained_position + 1300}]" if matching_trained_position is not None else "No match"
                    candidate_rows.append({
                        "enum": i,
                        "read": query,  # Read sequence
                        "read_reference_start_index": read_reference_start,  # Index where the read starts in the reference
                        "cigar_string": cigar_string,  # CIGAR string
                        "reference_interval": reference_interval,  # Reference interval
                        "is_exists_in_topk_fragments": 1,  # Is the read in the topk fragments
                        
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
                        "is_best_fragment_start_index_equal_to_index": is_best_frag_start_idx_eq_read_ref_start_idx,  # Is the best fragment start index equals to the read reference start index
                        
                        # Alignment string
                        "alignment_str": best_alignment_str,  # Best alignment string
                        "second_alignment_str": sorted_best_three_alignment_str[1],
                        "third_alignment_str": sorted_best_three_alignment_str[2],
                        "alignment_cigar_string": best_fragment_cigar_string,
                        "second_alignment_cigar_string": sorted_best_three_alignment_cigar_strings[1],
                        "third_alignment_cigar_string": sorted_best_three_alignment_cigar_strings[2],
                        
                        "gt_alignment_str": format_alignment(mapped_read.reference, query, read_reference_start-int(trained_pos), cigar_string) if is_best_frag_dist_aligns_best_sw_dist else None,
                    })
            # Set result to 1 if there's a matching index, 0 otherwise
            result = 1 if len(candidate_rows) > 0 else 0
            processed_results.append(result)
            
            # Determine reference interval
            reference_interval = f"[{matching_trained_position}, {matching_trained_position + 1300}]" if matching_trained_position is not None else "No match"
            if len(candidate_rows) > 0:
                best_row_based_on_distance = min(candidate_rows, key=lambda x: x["best_fragment_distance"])
                df_data.append(best_row_based_on_distance)
                
            else:
                df_data.append({
                        "enum": i,
                        "read": query,
                        "read_reference_start_index": read_reference_start,  # Index where the read starts in the reference
                        "cigar_string": "N/A" if not mapped_read else mapped_read.read.cigarstring,
                        "reference_interval": "No match",
                        "is_exists_in_topk_fragments": 0,  # Is the read in the topk fragments"
                    })
        else:
            processed_results.append(0)
            # Add row to dataframe data for reads with no matches
            df_data.append({
                "enum": enum,
                "read": query,
                "result": 0,
                "reference_start": "N/A",
                "reference_interval": "No match",
                "cigar_string": "N/A" if not mapped_read else mapped_read.read.cigarstring
            })
    
    # Create dataframe
    results_df = pd.DataFrame(df_data)
    
    return processed_results, results_df

def sv_results_refiner(results_path):
    eval_results = pd.read_csv(results_path)
    new_column_names = ["enum",
                    "read_gt", 
                    "read_gt_idx", 
                    "read_gt_cigar_str", 
                    "read/frag_gt_reference_interval", 
                    "is_read_gt_index_exist_in_top75", 
                    "read/frag_gt_top75_index", 
                    "is_read/frag_gt_top_match_in_top75", 
                    "read/frag_gt", 
                    "read/frag_gt_idx", 
                    "best_sw_score_in_top75", 
                    "read/frag_gt_best_sw_score",
                    "is_read/frag_gt_sw_score_best_in_top75",
                    "is_read/frag_gt_index_same_as_gt_index",
                    "alignment_str",
                    "second_best_alignment_str",
                    "third_best_alignment_str",
                    "alignment_cigar_str",
                    "second_best_alignment_cigar_str",
                    "third_best_alignment_cigar_str",
                    "read/frag_gt_alignment_str"]
    
    for idx, name in enumerate(new_column_names):
        eval_results.rename(columns={eval_results.columns[idx]: name}, inplace=True)
        
    eval_results["is_read_gt_index_exist_in_top75"] = eval_results["is_read_gt_index_exist_in_top75"].astype(bool)
    eval_results.loc[eval_results["is_read_gt_index_exist_in_top75"] == False, 
                 [
                 "read/frag_gt_top75_index", 
                 "is_read/frag_gt_top_match_in_top75", 
                 "read/frag_gt", 
                 "read/frag_gt_idx", 
                 "best_sw_score_in_top75", 
                 "read/frag_gt_best_sw_score",
                 "is_read/frag_gt_sw_score_best_in_top75",
                 "is_read/frag_gt_index_same_as_gt_index"]] = None
    
    eval_results["is_read/frag_gt_index_same_as_gt_index"] = None
    eval_results["is_read/frag_gt_index_same_as_gt_index"] = eval_results[eval_results["is_read/frag_gt_sw_score_best_in_top75"] == True].apply(lambda row: row["read_gt_idx"] == row["read/frag_gt_idx"], axis=1)
    eval_results["read_gt_idx_diff_from_read/frag_gt_index"] = eval_results["read_gt_idx"] - eval_results["read/frag_gt_idx"]
    
    return eval_results

def calculate_alignment_integrity(aligned_indices):
    """
    Calculates a structural integrity score for a pairwise alignment based on
    its aligned segment indices.

    The score favors alignments that are less fragmented (fewer chunks)
    and where the aligned portions densely cover the span of the alignment.
    Higher scores indicate better structural integrity. A score of ~2.0
    represents a single, perfect block alignment. Scores decrease as
    fragmentation increases or density decreases.

    Args:
        aligned_indices: A tuple containing two tuples of (start, end) indices
                         for target and query sequences, respectively, as
                         returned by Biopython's alignment.aligned property.
                         Format: (((t_start1, t_end1), ...), ((q_start1, q_end1), ...))

    Returns:
        A float score representing the structural integrity (higher is better),
        or 0.0 if the input is invalid or represents an empty alignment.

    Raises:
        ValueError: If the number of target chunks and query chunks differ.
    """
    # --- Input Validation ---
    if not isinstance(aligned_indices, np.ndarray) or len(aligned_indices) != 2:
        print("Warning: Input must be a tuple of two arrays.")
        return 0.0
    if not isinstance(aligned_indices[0], np.ndarray) or not isinstance(aligned_indices[1], np.ndarray):
         print("Warning: Input must be a tuple of two arrays.")
         return 0.0

    target_chunks = aligned_indices[0]
    query_chunks = aligned_indices[1]

    if len(target_chunks) != len(query_chunks):
        raise ValueError("Target and query chunk lists must have the same length.")

    N = len(target_chunks)
    if N == 0:
        return 0.0 # No alignment chunks, zero integrity

    # --- Calculate Total Aligned Length ---
    # Ensure start <= end and calculate length, summing across chunks
    total_target_aligned = sum(max(0, t_end - t_start) for t_start, t_end in target_chunks)
    total_query_aligned = sum(max(0, q_end - q_start) for q_start, q_end in query_chunks)

    # If total aligned length is zero, integrity is zero
    if total_target_aligned == 0 or total_query_aligned == 0:
        return 0.0

    # --- Calculate Alignment Span ---
    # Span is from the start of the first chunk to the end of the last chunk
    target_span_start = target_chunks[0][0]
    target_span_end = target_chunks[-1][1]
    query_span_start = query_chunks[0][0]
    query_span_end = query_chunks[-1][1]

    # Calculate span length, ensuring it's at least 1 to avoid division by zero
    target_span_length = max(1, target_span_end - target_span_start)
    query_span_length = max(1, query_span_end - query_span_start)

    # --- Calculate Alignment Density ---
    # Density = total aligned length within the span / length of the span
    target_density = total_target_aligned / target_span_length
    query_density = total_query_aligned / query_span_length

    # --- Calculate Final Integrity Score ---
    # Score = Sum of densities penalized by the number of chunks (N)
    # We divide by max(1, N) so N=1 isn't penalized, but N>1 is.
    integrity_score = (target_density + query_density) / max(1, N)

    return integrity_score

def clopper_pearson_interval(successes, trials, confidence_level=0.95):
    """
    Compute Error bounds using Clopper-Pearson method.
    """
    alpha = 1 - confidence_level

    lower_bound = stats.beta.ppf(alpha / 2, successes, trials - successes + 1)
    upper_bound = stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)

    return lower_bound, upper_bound

def get_embedding(
    model: EvalModel, sequences: List[str], batch_size=64
) -> torch.Tensor:
    """
    Given a model, a list of subsequences and a batch size, return the embeddings of the subsequences.

    Inputs:
        model: EvalModel, model to encode the subsequences
        sequences: list of strings, subsequences
        batch_size: int, batch size for encoding (default 64)

    Output:
        encodings: torch.Tensor, embeddings of the subsequences
    """

    encodings = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        batch_encodings = model.encode(batch)
        batch_encodings = torch.from_numpy(batch_encodings)
        encodings.append(batch_encodings)

    return torch.cat(encodings)


def get_closest_subsequence(
    subsequences: List[str], read: str, model: EvalModel, top_k: int = 1
) -> float:
    """

    Given the subsequences and the read, return the subsequence that has the highest cosine similarity with the read.

    Inputs:
        subsequence_embedding: torch.Tensor, embedding of the subsequence
        read_embedding: torch.Tensor, embedding of the read
        model: EvalModel, model to encode the subsequences
        top_k: int, number of top-k candidates to consider
    Output:
        cosine_similarity: float, cosine similarity between the subsequence and the read
    """

    read_embedding = get_embedding(model, [read])

    # Encode all subsequences in batches
    subsequence_encodings = get_embedding(model, subsequences)

    # Compute cosine similarities
    similarities = F.cosine_similarity(
        read_embedding.unsqueeze(0), subsequence_encodings
    )

    # Find the indices of the top k similarities
    top_k_similarities, top_k_indices = torch.topk(similarities, top_k)

    # Get the corresponding subsequences
    top_k_subsequences = [subsequences[i] for i in top_k_indices.tolist()]

    # Compute edit distances for top k subsequences
    edit_distances = [
        levenshtein_distance(read, subseq) for subseq in top_k_subsequences
    ]

    # Find the subsequence with the minimum edit distance
    min_edit_distance_index = np.argmin(edit_distances)
    best_subsequence = top_k_subsequences[min_edit_distance_index]
    best_similarity = top_k_similarities[min_edit_distance_index].item()
    best_edit_distance = edit_distances[min_edit_distance_index]

    return best_subsequence, best_similarity, best_edit_distance


def generate_subsequences(fragment: str, read_length: int) -> List[str]:
    """
    Given a fragment and a read, generate all subsequences of the fragment that are of the same length as the read.

    Input:
        fragment: string, fragment
        read_len: int, read
    Output:
        subsequences: list of strings, all subsequences of the fragment that are of the same length as the read
    """

    subsequences = []
    count = 0

    for i in range(len(fragment) - read_length + 1):
        subsequence = fragment[i : i + read_length]
        subsequences.append(subsequence)
        count += 1

    return subsequences, count

def get_sw_less_matching(
    all_candidate_strings: List[str], read: str, model: EvalModel
) -> Tuple[str, str, float]:
    """
    Given the top-k candidates and the read, split each candidate into a subsequence of length read, then compute
    model embedding for each of the subsequences and the read. Then compute cosine similarity between the read and the subsequences.
    Return the candidate with the highest cosine similarity.

    Inputs:
        all_candidate_strings: list of strings, top-k candidates
        read: string, read

    Outputs:
        candidate_fragment: string, candidate with the highest cosine similarity
        candidate_subsequence: string, subsequence of candidate with the highest cosine similarity
        candidate_score: float, cosine similarity between the read and the candidate with the highest cosine similarity
    """
    topk_subsequences = (
        []
    )  # Store all subsequences of the candidates returned by the vector store

    for candidate in all_candidate_strings:
        subsequences, _ = generate_subsequences(candidate, len(read))
        topk_subsequences.extend(subsequences)

    candidate_fragment, candidate_score, candidate_edit_distance = (
        get_closest_subsequence(topk_subsequences, read, model, 5)
    )

    return candidate_fragment, candidate_score, candidate_edit_distance


def align_real_reads(
    store,
    queries,
    top_k,
    distance_bound=0,
    batch_size=64,
    dictionary_of_values=None,
    meta=[],
):
    """
    Align real reads using the vector store.
    """

    num_queries = len(queries)
    finer_flag = np.zeros((num_queries, 1))

    for batch_start in tqdm(range(0, num_queries, batch_size)):
        batch_end = min(batch_start + batch_size, num_queries)
        batch_queries = queries[batch_start:batch_end]

        # TODO: Does not complete the search.

        returned = store.query_batch_real(batch_queries, top_k=top_k, meta=meta)

        for i, returned_unit in enumerate(returned):

            returned_unit_matched = returned_unit["matches"]
            trained_positions = [
                sample["metadata"]["position"] for sample in returned_unit_matched
            ]
            metadata_set = [
                sample["metadata"]["metadata"] for sample in returned_unit_matched
            ]
            all_candidate_strings = [
                sample["metadata"]["text"] for sample in returned_unit_matched
            ]

            (
                fragment_distances,
                _,
                _,
                _,
                sw_original,
                trained_positions,
                index_to_alignment_str,
                index_to_alignment_indices,
                _
            ) = bwamem_align_parallel(
                all_candidate_strings,
                trained_positions,
                metadata_set,
                returned_unit["query"],
                None,
            )

            if len(fragment_distances) != 0:

                is_within_score = is_within_score_bound(
                    fragment_distances,
                    distance_bound,
                    sw_original,
                )

                smallest_score = min(fragment_distances)
                if sw_original != -500:
                    print(
                        "Original SW distance is {} and min distance is {}".format(
                            sw_original, smallest_score
                        )
                    )

            else:
                is_within_score = 0

            if is_within_score:
                finer_flag[batch_start + i, 0] = 1
                # write into jsonlines file
                # with jsonlines.open(
                #     "test_cache/logs/pure_reads_SW_optimal", "a"
                # ) as writer:
                #     writer.write(
                #         {
                #             "smallest_SW_distance": smallest_score,
                #             "topk": top_k,
                #             "read": str(returned_unit["query"]),
                #         }
                #     )
            else:
                # Store information of failed aligment examples to run again with higher topk
                finer_flag[batch_start + i, 0] = 0

    return finer_flag[:num_queries, 0]


def get_alignment(
    returned_unit, dictionary_of_values, exactness, distance_bound, flex
):
    """Returns alignment details for a single query result."""
    returned_unit_matched = returned_unit["matches"]
    trained_positions = [sample["metadata"]["position"] for sample in returned_unit_matched]
    metadata_set = [sample["metadata"]["metadata"] for sample in returned_unit_matched]
    all_candidate_strings = [sample["metadata"]["text"] for sample in returned_unit_matched]
    original_sequence = dictionary_of_values.get(returned_unit["query"], None) if dictionary_of_values else None

    (
        fragment_distances,
        fragment_indices,
        distance_to_index,
        index_to_distance,
        sw_original,
        trained_positions,
        index_to_alignment_str,
        index_to_alignment_indices,
        _,
    ) = bwamem_align_parallel(
        all_candidate_strings,
        trained_positions,
        metadata_set,
        returned_unit["query"],
        original_sequence,
    )
    
    return {
        "query": returned_unit["query"],
        "fragments": all_candidate_strings,
        "distances": fragment_distances,
        "indices": fragment_indices,
        "original_sequence": original_sequence,
        "sw_original": sw_original,
        "distance_to_index": distance_to_index,
        "index_to_distance": index_to_distance,
        "trained_positions": trained_positions,
        "index_to_alignment_str": index_to_alignment_str,
    }

def get_best_alignment(
    returned_unit, dictionary_of_values, exactness, distance_bound, flex
):
    """Returns the best alignment result for a single query result."""
    returned_unit_matched = returned_unit["matches"]
    trained_positions = [sample["metadata"]["position"] for sample in returned_unit_matched]
    metadata_set = [sample["metadata"]["metadata"] for sample in returned_unit_matched]
    all_candidate_strings = [sample["metadata"]["text"] for sample in returned_unit_matched]
    
    original_sequence = dictionary_of_values.get(returned_unit["query"], None) if dictionary_of_values else None

    (
        fragment_distances,
        fragment_indices,
        distance_to_index,
        index_to_distance,
        sw_original,
        trained_positions,
        index_to_alignment_str,
        index_to_alignment_indices,
        _
    ) = bwamem_align_parallel(
        all_candidate_strings,
        trained_positions,
        metadata_set,
        returned_unit["query"],
        original_sequence,
    )
    distance_to_index_map = dict(zip(fragment_distances, fragment_indices))
    index_to_all_candidate_strings_map = dict(zip(fragment_indices, all_candidate_strings))
    best_distance = min(fragment_distances)
    best_index = distance_to_index_map[best_distance]
    best_fragment = index_to_all_candidate_strings_map[best_index]
    
    return {
        "query": returned_unit["query"],
        "best_fragment": best_fragment,
        "best_distance": best_distance,
        "best_index": best_index,
        "original_sequence": original_sequence,
        "sw_original": sw_original,
        "distance_to_index": distance_to_index,
        "index_to_distance": index_to_distance,
        "index_to_alignment_str": index_to_alignment_str,
    }

def log_mismatch(
    returned_unit, returned_index, smallest_score, ideal_index_score, ideal_index_fragment, distance_to_index, path_to_incorrect_index_file
):
    """Logs incorrect alignments for further review."""
    with jsonlines.open(path_to_incorrect_index_file, mode="a") as writer:
        min_sw_dist_fragments = [fragment for (_, _, _, fragment, _) in distance_to_index[smallest_score]]
        writer.write(
            {
                "read_index": returned_unit["index"],
                "best_index": returned_index,
                "smallest_SW_distance": smallest_score,
                "best_index_SW_distance": ideal_index_score,
                "read": str(returned_unit["query"]),
                "smallest_SW_fragments": min_sw_dist_fragments,
                "best_index_SW_fragments": ideal_index_fragment,
            }
        )

def flex_scoring(returned_unit, result, finer_flag, batch_start, i, exactness, distance_bound, path_to_incorrect_index_file, distributed, compare_type):
    """Handles scoring with flex mode."""
    is_within_location, returned_index = is_within_range_of_any_element(
        returned_unit["index"], result["indices"], exactness
    )
    is_within_score, returned_distance = is_within_score_bound(
        result["distances"], distance_bound, result["sw_original"]
    )
    
    try:
        smallest_distance = min(result["distances"])
    except:
        smallest_distance = -1
    
    if result['sw_original'] != -500:
        print(
            "Original SW distance is {} and min distance is {}".format(
                result["sw_original"], smallest_distance
            )
        )
        
    if returned_index:
        ideal_index_score, ideal_index_fragment = result["index_to_distance"][
            returned_index
        ]
        returned_index = int(returned_index)
        
    else:
        ideal_index_score, ideal_index_fragment = None, None
        
    if returned_distance:
        ideal_index_score, ideal_index_fragment = returned_distance, result["distance_to_index"][
            returned_distance
            ][0][3]
        
        returned_index = int(result["distance_to_index"][returned_distance][0][0]) + int(result["distance_to_index"][returned_distance][0][1])
    else:
        ideal_index_score, ideal_index_fragment, returned_index = smallest_distance, result["distance_to_index"][smallest_distance][0][3], int(result["distance_to_index"][smallest_distance][0][0]) + int(result["distance_to_index"][smallest_distance][0][1])
        
        if not distributed:
            log_mismatch(
                returned_unit,
                result["best_index"],
                result["best_distance"],
                ideal_index_score,
                ideal_index_fragment,
                distance_bound,
                result["indices"],
                path_to_incorrect_index_file,
            )
        
    if distributed:
        if compare_type == "both":
            if is_within_location and is_within_score:
                print("MATCH FOUND")
                finer_flag[batch_start + i, 0] = 1
            else:
                finer_flag[batch_start + i, 0] = 0
        elif compare_type == "location":
            if is_within_location:
                print("MATCH FOUND BY LOCATION")
                finer_flag[batch_start + i, 0] = 1
            else:
                finer_flag[batch_start + i, 0] = 0
        elif compare_type == "score":
            if is_within_score:
                print("MATCH FOUND BY SCORE")
                finer_flag[batch_start + i, 0] = 1
            else:
                finer_flag[batch_start + i, 0] = 0
    
    if not distributed:
        if is_within_location or is_within_score:
            finer_flag[batch_start + i, 0] = 1
        else:
            finer_flag[batch_start + i, 0] = 0
            
    # return is_within_location, is_within_score, returned_index, ideal_index_score

def normal_scoring(returned_unit, result, finer_flag, batch_start, i):
    """Handles normal scoring (non-flex mode)."""
    smallest_distance = min(result["distances"])
    if ((returned_unit["index"] in result["indices"])) or abs(
        smallest_distance + 2 * len(returned_unit["query"])
    ) < 1:
        finer_flag[batch_start + i, 0] = 1
    else:
        finer_flag[batch_start + i, 0] = 0

def score_alignment(successes, trials):
    """Computes confidence interval for alignment success rate."""
    lower_bound, upper_bound = clopper_pearson_interval(successes, trials)
    return lower_bound, upper_bound

def query_and_align(
    store, queries, indices, top_k, exactness=0, distance_bound=0, flex=False, per_k=0,
    batch_size=64, distributed=False, namespaces=None, namespace_dict=None, dictionary_of_values=None, return_type="best_alignment",
    compare_type="both"
):
    """Main function to align queries with stored sequences and return best alignments, all alignments, or just score."""
    work_dir = os.getcwd()
    path_to_incorrect_index_file = f"{work_dir}/evaluate/test_cache/logs/incorrect_index.jsonl"
    num_queries = len(queries)
    results = []
    finer_flag = np.zeros((num_queries, 1))
    
    for batch_start in tqdm(range(0, num_queries, batch_size)):
        batch_end = min(batch_start + batch_size, num_queries)
        batch_queries = queries[batch_start:batch_end]
        batch_indices = indices[batch_start:batch_end]
        
        if not distributed:
            returned = store.query_batch(batch_queries, batch_indices, top_k=top_k)
        else:
            if namespaces is None:
                returned = store.query_batch(batch_queries, 
                                             batch_indices,
                                             hotstart_list=None,
                                             meta_dict=None,
                                             prioritize=True,
                                             top_k=top_k)
            else:
                returned = store.query_batch(batch_queries, 
                                             batch_indices,  
                                             hotstart_list=namespaces[batch_start:batch_end], 
                                             meta_dict=namespace_dict, 
                                             prioritize=True,
                                             top_k=per_k)
        
        for i, returned_unit in enumerate(returned):
            if return_type == "alignment":
                result = get_alignment(
                    returned_unit, dictionary_of_values, exactness, distance_bound, flex
                )
            elif return_type == "best_alignment":
                result = get_best_alignment(
                    returned_unit, dictionary_of_values, exactness, distance_bound, flex
                )
            else:  # return_type == "score"
                result = get_alignment(
                    returned_unit, dictionary_of_values, exactness, distance_bound, flex
                )
                if flex:
                    flex_scoring(returned_unit, result, finer_flag, batch_start, i, exactness, distance_bound, path_to_incorrect_index_file, distributed, compare_type)
                else:
                    normal_scoring(returned_unit, result, finer_flag, batch_start, i)
                # continue #TODO: Uncomment this to skip appending to results if only score is needed
            
            
            results.append(result)
    
    if return_type == "score":
        lower_bound, upper_bound = score_alignment(np.sum(finer_flag), num_queries)
        return finer_flag[:num_queries, 0], lower_bound, upper_bound, results
    return results

def main_align(
    store,
    queries,
    indices,
    top_k,
    exactness=0,
    distance_bound=0,
    flex=False,
    batch_size=64,
    match=True,
    distributed=False,
    per_k=0,
    namespaces=None,
    namespace_dict=None,
    dictionary_of_values=None,
    compare_type = "both"
):

    """
    Main alignment function. Reads are batched and 
    """

    path_to_incorrect_index_file = (
        "/home/shreyas/NLP/dna2vec/evaluate/test_cache/logs" + "incorrect_index.jsonl"
    )

    if not distributed:

        num_queries = len(queries)
        finer_flag = np.zeros((num_queries, 1))

        for batch_start in tqdm(range(0, num_queries, batch_size)):
            batch_end = min(batch_start + batch_size, num_queries)
            batch_queries = queries[batch_start:batch_end]
            batch_indices = indices[batch_start:batch_end]

            returned = store.query_batch(batch_queries, batch_indices, top_k=top_k)  # 1
            # print(returned)
            for i, returned_unit in enumerate(returned):

                returned_unit_matched = returned_unit["matches"]
                trained_positions = [
                    sample["metadata"]["position"] for sample in returned_unit_matched
                ]
                metadata_set = [
                    sample["metadata"]["metadata"] for sample in returned_unit_matched
                ]
                all_candidate_strings = [
                    sample["metadata"]["text"] for sample in returned_unit_matched
                ]

                if dictionary_of_values is not None:
                    original_sequence = dictionary_of_values[returned_unit["query"]]
                else:
                    original_sequence = None

                (
                    fragment_distances,
                    fragment_indices,
                    distance_to_index,
                    index_to_distance,
                    sw_original,
                    trained_positions,
                    index_to_alignment_str,
                    index_to_alignment_indices,
                    timer,
                ) = bwamem_align_parallel(
                    all_candidate_strings,
                    trained_positions,
                    metadata_set,
                    returned_unit["query"],
                    original_sequence,
                )

                series = None

                if flex:

                    is_within_location, returned_index = is_within_range_of_any_element(
                        returned_unit["index"], fragment_indices, exactness
                    )

                    is_within_score = is_within_score_bound(
                        fragment_distances,
                        distance_bound,
                        sw_original,
                    )

                    smallest_score = min(fragment_distances)
                    if sw_original != -500:
                        print(
                            "Original SW distance is {} and min distance is {}".format(
                                sw_original, smallest_score
                            )
                        )

                    if returned_index:
                        ideal_index_score, ideal_index_fragment = index_to_distance[
                            returned_index
                        ]

                    if is_within_location and (ideal_index_score != smallest_score):
                        print(
                            "Mapped index distance {} does not match smallest distance {}".format(
                                ideal_index_score, smallest_score
                            )
                        )
                        # write into incorrect jsonl file

                        with jsonlines.open(
                            path_to_incorrect_index_file, mode="a"
                        ) as writer:

                            min_sw_dist_fragments = []
                            for (
                                starting_sub_index,
                                train_pos,
                                metadata,
                                fragment,
                                read,
                            ) in distance_to_index[smallest_score]:
                                min_sw_dist_fragments.append(fragment)

                            writer.write(
                                {
                                    "read_index": returned_unit["index"],
                                    "best_index": returned_index,
                                    "smallest_SW_distance": smallest_score,
                                    "best_index_SW_distance": ideal_index_score,
                                    "read": str(returned_unit["query"]),
                                    "smallest_SW_fragments": min_sw_dist_fragments,
                                    "best_index_SW_fragments": ideal_index_fragment,
                                }
                            )

                    if is_within_location or is_within_score:
                        print("MATCH FOUND")
                        finer_flag[batch_start + i, 0] = 1
                    else:
                        # Store information of failed aligment examples to run again with higher topk
                        finer_flag[batch_start + i, 0] = 0

                else:
                    if ((returned_unit["index"] in series) and match) or abs(
                        smallest_distance + 2 * len(returned_unit["query"])
                    ) < 1:
                        finer_flag[batch_start + i, 0] = 1
                    else:
                        finer_flag[batch_start + i, 0] = 0


        successes = np.sum(finer_flag)
        trials = num_queries
        lower_bound, upper_bound = clopper_pearson_interval(successes, trials)


        return finer_flag[:num_queries, 0], lower_bound, upper_bound

    else:  # recall - hotstart

        num_queries = len(queries)
        finer_flag = np.zeros((num_queries, 1))
        missed_queries = dict()

        for batch_start in tqdm(range(0, num_queries, batch_size)):
            batch_end = min(batch_start + batch_size, num_queries)
            batch_queries = queries[batch_start:batch_end]
            batch_indices = indices[batch_start:batch_end]

            # TODO: Does not complete the search.

            if namespaces is None:
                returned = store.query_batch(
                    batch_queries,
                    batch_indices,
                    hotstart_list=None,
                    meta_dict=None,
                    prioritize=True,
                    top_k=top_k,
                )
            else:
                returned = store.query_batch(
                    batch_queries,
                    batch_indices,
                    hotstart_list=namespaces[batch_start:batch_end],
                    meta_dict=namespace_dict,
                    prioritize=True,
                    top_k=per_k,
                )  # 1

            for i, returned_unit in enumerate(returned):

                returned_unit_matched = returned_unit["matches"]
                trained_positions = [
                    sample["metadata"]["position"] for sample in returned_unit_matched
                ]
                metadata_set = [
                    sample["metadata"]["metadata"] for sample in returned_unit_matched
                ]
                all_candidate_strings = [
                    sample["metadata"]["text"] for sample in returned_unit_matched
                ]

                if dictionary_of_values is not None:
                    original_sequence = dictionary_of_values[returned_unit["query"]]
                else:
                    original_sequence = None
                    
                (
                    fragment_distances,
                    fragment_indices,
                    distance_to_index,
                    index_to_distance,
                    sw_original,
                    trained_positions,
                    index_to_alignment_str,
                    index_to_alignment_indices,
                    timer,
                ) = bwamem_align_parallel(
                    all_candidate_strings,
                    trained_positions,
                    metadata_set,
                    returned_unit["query"],
                    original_sequence,
                )

                if flex:

                    is_within_location, returned_index = is_within_range_of_any_element(
                        returned_unit["index"], fragment_indices, exactness
                    )

                    is_within_score = is_within_score_bound(
                        fragment_distances,
                        distance_bound,
                        sw_original,
                    )

                    smallest_distance = min(fragment_distances)
                    if sw_original != -500:
                        print(
                            "Original SW distance is {} and min distance is {}".format(
                                sw_original, smallest_distance
                            )
                        )

                    if returned_index:
                        ideal_index_dist, ideal_index_fragment = index_to_distance[
                            returned_index
                        ]

                    if is_within_location and (ideal_index_dist != smallest_distance):
                        print(
                            "Mapped index distance {} does not match smallest distance {}".format(
                                ideal_index_dist, smallest_distance
                            )
                        )

                    if compare_type == "both":
                        if is_within_location and is_within_score:
                            print("MATCH FOUND")
                            finer_flag[batch_start + i, 0] = 1
                        else:
                            finer_flag[batch_start + i, 0] = 0
                    elif compare_type == "location":
                        if is_within_location:
                            print("MATCH FOUND BY LOCATION")
                            finer_flag[batch_start + i, 0] = 1
                        else:
                            finer_flag[batch_start + i, 0] = 0
                    elif compare_type == "score":
                        if is_within_score:
                            print("MATCH FOUND BY SCORE")
                            finer_flag[batch_start + i, 0] = 1
                        else:
                            finer_flag[batch_start + i, 0] = 0

                else:
                    if ((returned_unit["index"] in series) and match) or abs(
                        smallest_distance + 2 * len(returned_unit["query"])
                    ) < 1:
                        finer_flag[batch_start + i, 0] = 1
                    else:
                        finer_flag[batch_start + i, 0] = 0


        successes = np.sum(finer_flag)
        lower_bound, upper_bound = clopper_pearson_interval(successes, num_queries)


        return finer_flag[:num_queries, 0], lower_bound, upper_bound


def read_fasta_chromosomes(file_path):
    """
    Generator function to read a FASTA file and extract each chromosome sequence with its header.

    Parameters:
        file_path (str): Path to the FASTA file.

    Yields:
        tuple: A tuple containing the chromosome header and sequence data.
    """

    import os

    with open(file_path, "r") as file:
        header = None
        sequence = ""
        for line in tqdm(file):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            if line.startswith(">"):

                # If the line starts with '>', it is a chromosome header
                if header is not None:
                    # Check if folder exists
                    if not os.path.exists("test_cache/logs"):
                        os.makedirs("test_cache/logs")

                    with open("test_cache/logs/headers", "a+") as f:
                        f.write(header)
                        f.write("\n")
                    yield (header, sequence)

                header = line  # Extract the header without '>'
                sequence = ""

            else:
                sequence += line
        # Yield the last chromosome entry in the file
        if header is not None:
            if not os.path.exists("test_cache/logs"):
                os.makedirs("test_cache/logs")
            with open("test_cache/logs/headers", "a+") as f:
                f.write(header)
                f.write("\n")
            yield (header, sequence)


def is_within_range_of_any_element(X: int, Y: list, exactness: int):
    """
    Check if the read is within the range of the fragment
    """
    for element in Y:
        if abs(X - element) <= exactness:
            return True, element

    return False, None


def is_within_score_bound(
    read_sw_distances: list, score_bound: int, sw_original: int
):
    """
    Check if smith waterman score between fragment and read is within the score bound
    """

    for dist in read_sw_distances:
        if sw_original is not None:
            if dist - sw_original <= score_bound:
                return True , dist

    return False, None #TODO: Change this to return the distance


def initialize_pinecone(
    checkpoint_queue: list[str], data_queue: list[str], device: str, pod_type: str
):
    """
    Inputs : checkpoint_queue -> model checkpoints.
                data_queue -> data sources.
                device -> device to run the model on.
                pod_type -> pod type to run the Pinecone index on.

    Outputs: Yields a PineconeStore object, data_alias and config.
    """
    
    for alias in checkpoint_queue:
        
        if alias == "huggingface":
            print("Huggingface Model")
            print("______________________")
            model, tokenizer, pooling = load_hf_model()
            model_params = {
                "tokenizer": tokenizer,
                "model": model,
                "pooling": pooling,
            }
            baseline = False
            baseline_name = None
            hf_model = True
            hf_model_name = alias
        else:
            # Check if provided alias is in the models trained and not baseline.
            if alias in configs["checkpoints"] and configs["checkpoints"][alias] != "Baseline":
                print(f"{alias} Model")
                print("______________________")
                received = torch.load(configs["checkpoints"][alias], map_location="cpu")
                config = received["config"]
                model_config = config.model_config.model_dump()
                model_config["tokenizer_path"] = configs["checkpoints"]["tokenizer"]
                model_config = ModelConfigSchema(**model_config)
                encoder, pooling, tokenizer = model_from_config(model_config)
                encoder.load_state_dict(received["model"])
                encoder.eval()
                model_params = {
                    "tokenizer": tokenizer,
                    "model": encoder,
                    "pooling": pooling,
                }
                baseline = False
                baseline_name = None
                hf_model = False
                hf_model_name = None

            # Check if model is baseline
            elif alias in configs["checkpoints"] and configs["checkpoints"][alias] == "Baseline":
                model_params = None
                baseline = True
                baseline_name = alias
                hf_model = False
                hf_model_name = None

        for data_alias in data_queue:
            config = str("config-" + alias + "-" + data_alias).lower()
            store = PineconeStore(
                device=torch.device(device),
                index_name=str(
                    "config-" + alias + "-" + data_alias.replace(",", "-")
                ).lower(),
                metric="cosine",
                model_params=model_params,
                baseline_name=baseline_name,
                baseline=baseline,
                hf_model=hf_model,
                hf_model_name=hf_model_name,
                pod_type=pod_type,
            )
            yield store, data_alias, config


def sample_subsequence(string: str, min_length: int = 150, max_length: int = 350):

    subseq_length = random.randint(min_length, max_length)
    # Generate a random starting index
    start_index = random.randint(0, len(string) - subseq_length)
    # Extract the subsequence
    subsequence = string[start_index : start_index + subseq_length]
    return subsequence


def pick_random_lines(
    path: str = "/home/pholur/dna2vec/tests/data/subsequences_sample_train.txt",
    k=100,
    mode: str = "random",
    sequences_prior: int = 0,
):
    import pickle

    with open(path, "rb") as file:
        units = pickle.load(file)

    if mode == "random":
        random_lines = random.sample(units, k)
        return random_lines

    elif mode == "sequenced":
        random_index = random.sample(range(len(units) - k), 1)[0]
        random_lines = units[random_index : random_index + k]
        return random_lines

    elif mode == "subsequenced":

        if k % sequences_prior != 0:
            print(
                "k does not divide perfetly by sequences_prior resulting in fewer samples."
            )

        random_lines = []
        real_k = k // sequences_prior
        random_indices = random.sample(range(len(units)), sequences_prior)
        for random_index in random_indices:
            line = units[random_index]
            random_lines.append(line)

            for _ in range(real_k):
                random_lines.append(
                    (sample_subsequence(line["text"]), str(random_index))
                )
        return random_lines

    else:
        raise NotImplementedError("Mode not defined.")


def pick_from_special_gene_list(
    gene_path="/home/pholur/dna2vec/tests/data/ch2_genes.csv",
    full_path="/home/pholur/dna2vec/tests/data/NC_000002.12.txt",
    samples=5000,
):
    import pandas as pd

    df = pd.read_csv(gene_path)

    with open(full_path, "r") as f:
        characters = f.read()

    sequences = []
    samples_per = samples // df.shape[0]

    for _, row in df.iterrows():

        label = row["name"]
        indices = row["indices"].split(";")
        big_sequence = characters[int(indices[0]) : int(indices[1])]

        big_sequence = big_sequence[
            len(big_sequence) // 2 - 500 : len(big_sequence) // 2 + 500
        ]
        t = 0
        sequences.append((big_sequence, label + "_anchor"))

        for _ in range(samples_per):
            sequences.append((sample_subsequence(big_sequence), label))

    return sequences


def pick_from_chimp_2a_2b(
    path_2a: str, path_2b: str, samples: int, per_window: int = 1000
):

    per_region = samples // 2
    number_of_cuts = per_region // per_window

    with open(path_2a, "r") as f:
        chromosome_2a = f.read()

    with open(path_2b, "r") as f:
        chromosome_2b = f.read()

    def return_sequences_from_chimp(text_sequence: str, label: str):
        random_lines = []
        random_indices = random.sample(
            range(len(text_sequence) - per_window), number_of_cuts
        )
        for random_index in random_indices:
            full_sequence = text_sequence[random_index : random_index + per_window]
            for _ in range(per_window):
                random_lines.append((sample_subsequence(full_sequence), label))
        return random_lines

    full_sequences = return_sequences_from_chimp(chromosome_2a, "chimp_2a")
    full_sequences.extend(return_sequences_from_chimp(chromosome_2b, "chimp_2b"))

    return full_sequences


def pick_from_chromosome3(path, samples, per_window=1000):
    number_of_cuts = samples // per_window

    with open(path, "r") as f:
        chromosome_3 = f.read()

    random_lines = []
    random_indices = random.sample(
        range(len(chromosome_3) - per_window), number_of_cuts
    )
    for random_index in random_indices:
        full_sequence = chromosome_3[random_index : random_index + per_window]
        random_lines.append((full_sequence, "anchor_" + str(random_index)))

        for _ in range(per_window):
            random_lines.append((sample_subsequence(full_sequence), str(random_index)))

    return random_lines


def calculate_SW_and_Cosine_similarity(
    read: str,
    fragment: str,
):
    """
    Calculate Smith-Waterman similarity and cosine similarity between a read and a fragment.
    """
    def load_hf_model():
        hf_model = AutoModel.from_pretrained("roychowdhuryresearch/dna2vec", trust_remote_code=True)
        hf_tokenizer = AutoTokenizer.from_pretrained("roychowdhuryresearch/dna2vec", trust_remote_code=True)

        class AveragePooler(nn.Module):
            """
            Parameter-free poolers to get the sentence embedding
            # derived from https://github.com/princeton-nlp/SimCSE/blob/13361d0e29da1691e313a94f003e2ed1cfa97fef/simcse/models.py#LL49C1-L84C1
            """

            def __init__(self):
                super().__init__()

            def forward(self, last_hidden, attention_mask):
                # Old previous implementation
                return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
                    -1
                ).unsqueeze(-1)

        hf_model.pooler = AveragePooler()
        return hf_model, hf_tokenizer, hf_model.pooler
    
    hf_model, hf_tokenizer, hf_model.pooler = load_hf_model()
    
    # Calculate Smith-Waterman similarity
    sw_score = calculate_smith_waterman_distance(read, fragment)[["distance"]]
    begins = calculate_smith_waterman_distance(read, fragment)[["begins"]]

    # Calculate cosine similarity of the embeddings
    read_embedding = hf_model.pooler(hf_model(**hf_tokenizer(read, return_tensors="pt")), hf_tokenizer(read, return_tensors="pt").attention_mask)
    fragment_embedding = hf_model.pooler(hf_model(**hf_tokenizer(fragment, return_tensors="pt")), hf_tokenizer(fragment, return_tensors="pt").attention_mask)
    cosine_score = torch.nn.functional.cosine_similarity(read_embedding, fragment_embedding).item()

    return sw_score, begins, cosine_score
    
    

def format_alignment(target_seq, query_seq, target_start, cigar_string=None, alignment_indices=None, line_width=60):
    """
    Format the alignment between target and query sequences based on CIGAR string or alignment indices.
    
    Args:
        target_seq: The raw target/reference sequence (without gaps)
        query_seq: The raw query sequence (without gaps)
        target_start: The starting position of the target sequence in the reference
        cigar_string: Optional; the CIGAR string representing the alignment
        alignment_indices: Optional; tuple of target and query alignment indices
        line_width: Width of each line in the alignment display (default: 60)
        
    Returns:
        A formatted string showing the alignment
    """
    # First, insert gaps in the sequences according to alignment information
    if cigar_string is not None:
        gapped_target, gapped_query, match_line = process_cigar(target_seq, query_seq, cigar_string)
    elif alignment_indices is not None:
        gapped_target, gapped_query, match_line = process_alignment_indices(target_seq, query_seq, alignment_indices)
    else:
        raise ValueError("Either cigar_string or alignment_indices must be provided")
    
    # Format the alignment in blocks
    result = []
    target_pos = target_start
    query_pos = 0
    
    for i in range(0, len(gapped_target), line_width):
        block_target = gapped_target[i:i+line_width]
        block_match = match_line[i:i+line_width]
        block_query = gapped_query[i:i+line_width]
        
        # Calculate displayed positions
        target_displayed_chars = sum(1 for c in block_target if c != '-')
        query_displayed_chars = sum(1 for c in block_query if c != '-')
        
        # Use consistent padding for better alignment
        result.append(f"target {target_pos:10d} {block_target}")
        result.append(f"       {i:10d} {block_match}")
        result.append(f"query  {query_pos:10d} {block_query}")
        result.append("")
        
        # Update positions for next block
        target_pos += target_displayed_chars
        query_pos += query_displayed_chars
    
    return "\n".join(result)

def process_cigar(target_seq, query_seq, cigar_string):
    """Process CIGAR string to insert gaps in sequences and create match line."""
    import re
    
    cigar_ops = re.findall(r'(\d+)([MIDNSHP=X])', cigar_string)
    
    gapped_target = []
    gapped_query = []
    match_line = []
    
    t_idx = q_idx = 0
    
    for length, op in cigar_ops:
        length = int(length)
        
        if op == 'M':  # Match or mismatch
            for i in range(length):
                if t_idx < len(target_seq) and q_idx < len(query_seq):
                    gapped_target.append(target_seq[t_idx])
                    gapped_query.append(query_seq[q_idx])
                    match_line.append('|' if target_seq[t_idx] == query_seq[q_idx] else '.')
                    t_idx += 1
                    q_idx += 1
        elif op == 'I':  # Insertion in query
            for i in range(length):
                if q_idx < len(query_seq):
                    gapped_target.append('-')
                    gapped_query.append(query_seq[q_idx])
                    match_line.append('-')
                    q_idx += 1
        elif op == 'D':  # Deletion in query
            for i in range(length):
                if t_idx < len(target_seq):
                    gapped_target.append(target_seq[t_idx])
                    gapped_query.append('-')
                    match_line.append('-')
                    t_idx += 1
        elif op == 'S':  # Soft clipping
            for i in range(length):
                if q_idx < len(query_seq):
                    # Soft clips are in query but not aligned to target
                    gapped_target.append('-')
                    gapped_query.append(query_seq[q_idx])
                    match_line.append(' ')
                    q_idx += 1
    
    return ''.join(gapped_target), ''.join(gapped_query), ''.join(match_line)

def process_alignment_indices(target_seq, query_seq, alignment_indices):
    """Process alignment indices to insert gaps in sequences and create match line."""
    target_indices, query_indices = alignment_indices
    
    # Create arrays to track which positions are aligned
    target_aligned = [False] * len(target_seq)
    query_aligned = [False] * len(query_seq)
    
    # Mark aligned positions
    for i in range(len(target_indices)):
        t_start, t_end = target_indices[i]
        q_start, q_end = query_indices[i]
        
        for j in range(min(t_end - t_start, q_end - q_start)):
            target_aligned[t_start + j] = True
            query_aligned[q_start + j] = True
    
    # Build gapped sequences
    gapped_target = []
    gapped_query = []
    match_line = []
    
    t_idx = q_idx = 0
    
    # Process soft clips at the beginning
    if not query_aligned[0] and len(query_indices) > 0 and query_indices[0][0] > 0:
        for i in range(query_indices[0][0]):
            gapped_target.append('-')
            gapped_query.append(query_seq[i])
            match_line.append('.')
            q_idx = query_indices[0][0]
    
    # Process aligned regions and gaps between them
    for i in range(len(target_indices)):
        t_start, t_end = target_indices[i]
        q_start, q_end = query_indices[i]
        
        # Add any gap between last chunk and this one
        if i > 0:
            last_t_end = target_indices[i-1][1]
            last_q_end = query_indices[i-1][1]
            
            # Add unaligned target sequence (deletion)
            for j in range(last_t_end, t_start):
                gapped_target.append(target_seq[j])
                gapped_query.append('-')
                match_line.append('.')
                
            # Add unaligned query sequence (insertion)
            for j in range(last_q_end, q_start):
                gapped_target.append('-')
                gapped_query.append(query_seq[j])
                match_line.append('.')
        
        # Add the aligned chunk
        for j in range(min(t_end - t_start, q_end - q_start)):
            gapped_target.append(target_seq[t_start + j])
            gapped_query.append(query_seq[q_start + j])
            match_line.append('|' if target_seq[t_start + j] == query_seq[q_start + j] else '.')
    
    # Process soft clips at the end
    if len(query_indices) > 0 and query_indices[-1][1] < len(query_seq):
        for i in range(query_indices[-1][1], len(query_seq)):
            gapped_target.append('-')
            gapped_query.append(query_seq[i])
            match_line.append('.')
    
    return ''.join(gapped_target), ''.join(gapped_query), ''.join(match_line)
    
    
