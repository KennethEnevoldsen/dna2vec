import random
import yaml

from pinecone_store import PineconeStore

from alignment_metrics import calculate_smith_waterman_distance
from collections import defaultdict

# import sys
# sys.path.append("../src/")
from dna2vec.model import model_from_config

from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import time


with open("configs/data_recipes.yaml", 'r') as stream:
    data_recipes = yaml.safe_load(stream)
    
with open("configs/model_checkpoints.yaml", 'r') as stream:
    checkpoints = yaml.safe_load(stream)
    
with open("configs/raw.yaml", 'r') as stream:
    raw_fasta_files = yaml.safe_load(stream)


def initialize_pinecone(checkpoint_queue: list[str], 
                        data_queue: list[str], 
                        device:str="cuda:0"
):
    
    import torch
    
    for alias in checkpoint_queue:
        
        if alias in checkpoints:
            received = torch.load(checkpoints[alias])
        else:
            received = torch.load(alias)
            
        config = received["config"]
        config.model_config.tokenizer_path = checkpoints["tokenizer"]
        encoder, pooling, tokenizer = model_from_config(config.model_config)
        encoder.load_state_dict(received["model"])
        encoder.eval()
        
        for data_alias in data_queue:
            config = str("config-" + alias + "-" + data_alias).lower()
            store = PineconeStore(
                                    device = torch.device(device),
                                    index_name = str("config-" + alias + "-" + data_alias.replace(",","-")).lower(),
                                    metric = "cosine",
                                    model_params = {
                                        "tokenizer": tokenizer,
                                        "model": encoder,
                                        "pooling": pooling,
                                    }
                                )
            
            yield store, data_alias, config


def sample_subsequence(string: str, 
                       min_length: int = 150, 
                       max_length: int = 350
):
    
    subseq_length = random.randint(min_length, max_length)
    # Generate a random starting index
    start_index = random.randint(0, len(string) - subseq_length)
    # Extract the subsequence
    subsequence = string[start_index : start_index + subseq_length]
    return subsequence


def pick_random_lines(path:str = "/home/pholur/dna2vec/tests/data/subsequences_sample_train.txt",
                      k = 100,
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
        random_lines = units[random_index:random_index+k]
        return random_lines
    
    elif mode == "subsequenced":

        if k % sequences_prior != 0:
            print("k does not divide perfetly by sequences_prior resulting in fewer samples.")
        
        random_lines = []
        real_k = k // sequences_prior
        random_indices = random.sample(range(len(units)), sequences_prior)
        for random_index in random_indices:
            line = units[random_index]
            random_lines.append(line)

            for _ in range(real_k):
                random_lines.append((sample_subsequence(line["text"]), str(random_index)))
        return random_lines

    else:
        raise NotImplementedError("Mode not defined.")


def pick_from_special_gene_list(
            gene_path = "/home/pholur/dna2vec/tests/data/ch2_genes.csv",
            full_path = "/home/pholur/dna2vec/tests/data/NC_000002.12.txt",
            samples = 5000,
):
    import pandas as pd
    df = pd.read_csv(gene_path)
    
    with open(full_path, "r") as f:
        characters = f.read()
    
    sequences = []
    samples_per = samples // df.shape[0]
    
    for _,row in df.iterrows():
        
        label = row["name"]
        indices = row["indices"].split(";")
        big_sequence = characters[int(indices[0]): int(indices[1])]
        
        big_sequence = big_sequence[len(big_sequence) // 2  - 500: len(big_sequence) // 2  + 500]
        t = 0
        sequences.append((big_sequence, label + "_anchor"))

        for _ in range(samples_per):
            sequences.append((sample_subsequence(big_sequence), label))
    
    return sequences


def pick_from_chimp_2a_2b(path_2a: str, 
                          path_2b: str, 
                          samples: int, 
                          per_window: int = 1000
):
    
    per_region = samples // 2
    number_of_cuts = per_region // per_window
    
    with open(path_2a, "r") as f:
        chromosome_2a = f.read()
        
    with open(path_2b, "r") as f:
        chromosome_2b = f.read()
    
    def return_sequences_from_chimp(text_sequence: str, 
                                    label: str
    ):
        random_lines = []
        random_indices = random.sample(range(len(text_sequence) - per_window), number_of_cuts)
        for random_index in random_indices:
            full_sequence = text_sequence[random_index:random_index + per_window]
            for _ in range(per_window):
                random_lines.append((sample_subsequence(full_sequence), label))
        return random_lines
    
    full_sequences = return_sequences_from_chimp(chromosome_2a, "chimp_2a")
    full_sequences.extend(return_sequences_from_chimp(chromosome_2b, "chimp_2b"))
    
    return full_sequences


def pick_from_chromosome3(path, samples, per_window = 1000):
    number_of_cuts = samples // per_window
    
    with open(path, "r") as f:
        chromosome_3 = f.read()

    random_lines = []
    random_indices = random.sample(range(len(chromosome_3) - per_window), number_of_cuts)
    for random_index in random_indices:
        full_sequence = chromosome_3[random_index:random_index + per_window]
        random_lines.append((full_sequence, "anchor_" + str(random_index)))

        for _ in range(per_window):
            random_lines.append((sample_subsequence(full_sequence), str(random_index)))
            
    return random_lines


def bwamem_align(all_candidate_strings: list[str], 
                 trained_positions: list[str], 
                 metadata_set: list[str], 
                 substring: list[str]
):
    '''
    Currently implemented sequentially
    '''
    
    total_time = 0
    refined_results = defaultdict(list)
    
    for long_string, train_pos, metadata in zip(
        all_candidate_strings, trained_positions, metadata_set):
        
        returned_object = calculate_smith_waterman_distance(long_string, substring)
        total_time += returned_object["elapsed time"]

        for starting_sub_index in returned_object["begins"]:
            refined_results[returned_object["distance"]].append(
                (starting_sub_index, train_pos, metadata)
            )
    
    try:
        smallest_key = min(refined_results.keys())
        
    except ValueError:
        return [], [], [], total_time
    
    smallest_values = refined_results[smallest_key]
    
    identified_sub_indices = []
    identified_indices = []
    metadata_indices = []
    
    for term in smallest_values:
        identified_sub_indices.append(term[0])
        identified_indices.append(term[1])
        metadata_indices.append(term[2])      
          
    return identified_sub_indices, identified_indices, metadata_indices, total_time


def process_single_string(args:tuple):
    long_string, substring, train_pos, metadata = args
    returned_object = calculate_smith_waterman_distance(long_string, substring)
    return returned_object["distance"], returned_object["begins"], train_pos, metadata, returned_object["elapsed time"]


def bwamem_align_parallel(all_candidate_strings: list[str], 
                          trained_positions: list[str], 
                          metadata_set: list[str], 
                          substring: str, 
                          max_workers: int=50
):
    
    total_time = time.time()
    refined_results = defaultdict(list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_string, 
                                    zip(all_candidate_strings, 
                                        [substring]*len(all_candidate_strings), 
                                        trained_positions, 
                                        metadata_set)))

    for distance, begins, train_pos, metadata, _ in results:
        for starting_sub_index in begins:
            refined_results[distance].append(
                (starting_sub_index, train_pos, metadata)
            )
    
    try:
        smallest_key = min(refined_results.keys())
    except ValueError:
        return [], [], [], time.time() - total_time
    
    smallest_values = refined_results[smallest_key]
    
    identified_sub_indices = []
    identified_indices = []
    metadata_indices = []
    
    for term in smallest_values:
        identified_sub_indices.append(term[0])
        identified_indices.append(term[1])
        metadata_indices.append(term[2])      
          
    return identified_sub_indices, identified_indices, metadata_indices, time.time() - total_time


