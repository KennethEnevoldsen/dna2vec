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

# Local imports
from aligners.smith_waterman import bwamem_align_parallel
from dna2vec.model import model_from_config
from inference_models import EvalModel
from pinecone_store import PineconeStore

from transformers import AutoModel, AutoTokenizer
import torch.nn as nn


# Define the configuration files
config_files = {
    "data_recipes": "/home/yigit/codebase/dna2vec/evaluate/configs/data_recipes.yaml",
    "checkpoints": "/home/yigit/codebase/dna2vec/evaluate/configs/model_checkpoints.yaml",
    "raw_fasta_files": "/home/yigit/codebase/dna2vec/evaluate/configs/raw.yaml"
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
                _,
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
                with jsonlines.open(
                    "test_cache/logs/pure_reads_SW_optimal", "a"
                ) as writer:
                    writer.write(
                        {
                            "smallest_SW_distance": smallest_score,
                            "topk": top_k,
                            "read": str(returned_unit["query"]),
                        }
                    )
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
        _,
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
    is_within_score = is_within_score_bound(
        result["distances"], distance_bound, result["sw_original"]
    )
    
    smallest_distance = min(result["distances"])
    
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
        
    if is_within_location and (ideal_index_score != smallest_distance):
        print(
            "Mapped index distance {} does not match smallest distance {}".format(
                ideal_index_score, smallest_distance
            )
        )
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
    path_to_incorrect_index_file = "/home/yigit/codebase/dna2vec/evaluate/test_cache/logs/incorrect_index.jsonl"
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
                continue  # Skip appending to results if only score is needed
            
            results.append(result)
    
    if return_type == "score":
        lower_bound, upper_bound = score_alignment(np.sum(finer_flag), num_queries)
        return finer_flag[:num_queries, 0], lower_bound, upper_bound
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
                return True

    return False


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
                received = torch.load(configs["checkpoints"][alias], map_location="cpu")
                config = received["config"]
                config.model_config.tokenizer_path = configs["checkpoints"]["tokenizer"]
                encoder, pooling, tokenizer = model_from_config(config.model_config)
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
