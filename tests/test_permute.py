import sys
sys.path.append("../src/")

import torch
from tqdm import tqdm
import yaml
import argparse
from pinecone_store import PineconeStore
from dna2vec.model import model_from_config
from helpers import pick_random_lines, initialize_pinecone, data_recipes, checkpoints
import numpy as np
from typing import Literal
import random


parser = argparse.ArgumentParser(description="Permute evaluate")
parser.add_argument('--recipes', type=str)
parser.add_argument('--checkpoints', type=str)
parser.add_argument('--mode', type=str)

    


def edit(query, edit_mode, generalize):
    
    if edit_mode == "end deplete":
        if np.random.rand() > 0.5:
            return query[generalize:]
        else:
            return query[:-1*generalize]
        
    elif edit_mode == "swap":
        char_list = list(query)
        idx = list(range(len(char_list)))
        
        for _ in range(generalize):
            i1, i2 = random.sample(idx, 2)
            char_list[i1], char_list[i2] = char_list[i2], char_list[i1]
        
        return ''.join(char_list)

    elif edit_mode == "random deplete":
        try:
            start_index = random.sample(range(len(query) - generalize), 1)[0]
            query = query[:start_index] + query[(start_index + generalize):]
        except:
            query = query
        return query

    else:
        raise NotImplementedError("Edit mode is under-defined.")




def permute(store, query, index, top_k, metadata, edit_mode, generalize):
    
    train_flag = np.zeros((1,len(query)))
    start = 0
    for _ in range(len(query) // generalize):
        
        returned = store.query([query], top_k=top_k)["matches"]
        trained_positions = {sample["metadata"]["position"] for sample in returned}
        metadata_set = {sample["metadata"]["metadata"] for sample in returned}
        
        if index in trained_positions and metadata in metadata_set:
            train_flag[0,start:start + generalize] = 1
        else:
            train_flag[0,start:start + generalize] = 0
        
        start += generalize
        query = edit(query, edit_mode, generalize)
        
    return train_flag



def identify_random_subsequence(query, length):
    if length > len(query):
        return query  # Return None if the requested length is longer than the query
    start_index = random.randint(0, len(query) - length)  # Generate a random starting index within the valid range
    end_index = start_index + length  # Calculate the end index
    return start_index, query[start_index:end_index]


from alignment_metrics import calculate_smith_waterman_distance
from collections import defaultdict



def bwamem_align(all_candidate_strings, trained_positions, metadata_set, substring):
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
        



import time
def custom_random_sub(store, query, index, top_k, metadata, generalize):
    train_flag = np.zeros((1,len(query)))
    finer_flag = np.zeros_like(train_flag)
    timer_flag = np.zeros_like(train_flag)
    total_timer_flag = np.zeros_like(train_flag)
    start = 0

    for k in tqdm(range(1, len(query) // generalize)):
        
        beginning_time = time.time()

        sub_index, substring = identify_random_subsequence(
            query, min(generalize*k, len(query)))
        
        returned = store.query([substring], top_k=top_k)["matches"]

        trained_positions = [sample["metadata"]["position"] for sample in returned]
        metadata_set = [sample["metadata"]["metadata"] for sample in returned]
        
        if (metadata,index) in zip(metadata_set, trained_positions):
            train_flag[0,start:start + generalize] = 1
        else:
            train_flag[0,start:start + generalize] = 0
        
        all_candidate_strings = [sample["metadata"]["text"] for sample in returned]
        
        identified_sub_indices, identified_indices, metadata_indices, timer = bwamem_align(
                                                                                            all_candidate_strings, 
                                                                                            trained_positions, 
                                                                                            metadata_set,
                                                                                            substring)
        
        # print((sub_index, index, metadata), list(zip(
        #     identified_sub_indices, identified_indices, metadata_indices)))
        # exit()
        
        if (sub_index, index, metadata) in zip(
            identified_sub_indices, identified_indices, metadata_indices):
            finer_flag[0,start:start + generalize] = 1
        else:
            finer_flag[0,start:start + generalize] = 0
        timer_flag[0, start:start + generalize] = timer
        total_timer_flag[0, start:start + generalize] = time.time() - beginning_time
        start += generalize
      
    return train_flag, finer_flag, timer_flag, total_timer_flag


def main(paths:list,
         store: PineconeStore = None,
         config: str = None,
         test_k: int = 200,
         top_k: int = 50,
         generalize: int = 5,
         edit_mode: Literal["random deplete", "end deplete", "swap", "random_sub"] = "random_sub"
):
    
    per_samples = test_k // len(paths)
    test_lines = []
    
    for path in paths:
        test_lines.extend(pick_random_lines(path, per_samples))
    
    train_flags = np.zeros((test_k, len(test_lines[0]["text"])))   
    finer_flags = np.zeros_like(train_flags)
    timer_flags = np.zeros_like(train_flags)
    total_timer_flags = np.zeros_like(train_flags)
    
    for i,line in tqdm(enumerate(test_lines)):
        
        query = line["text"]
        index = line["position"]
        metadata = line["metadata"]
        
        if edit_mode == "random_sub":
            train_flag, finer_flag, timer, total_timer = custom_random_sub(store, 
                                                       query, 
                                                       index, 
                                                       top_k,  
                                                       metadata,
                                                       generalize)
            finer_flags[i,:] = finer_flag
            timer_flags[i,:] = timer
            total_timer_flags[i,:] = total_timer
            
        else:
            train_flag = permute(store, 
                                query, 
                                index, 
                                top_k,  
                                metadata, 
                                edit_mode, 
                                generalize)
            
        train_flags[i,:] = train_flag
        
    np.savez_compressed(f"test_cache/permute/run_\
{config}_{edit_mode}_{test_k}_{top_k}_{generalize}.npz", train = train_flags, finer = finer_flags, timer = timer_flags, total_timer = total_timer_flags)
        



if __name__ == "__main__":

    args = parser.parse_args()
    
    data_queue = args.recipes.split(";")
    checkpoint_queue = args.checkpoints.split(";")
    
    for (store, data_alias, config) in initialize_pinecone(checkpoint_queue, data_queue):
        list_of_data_sources = []
        sources = data_alias.split(",")
        for source in sources:
            if source in data_recipes:
                list_of_data_sources.append(data_recipes[source])
            else:
                list_of_data_sources.append(source)
                
        main(list_of_data_sources, store, config, top_k=5, edit_mode=args.mode, test_k = 500, generalize=25)
        # main(list_of_data_sources, store, config, top_k=50, edit_mode=args.mode, test_k = 1000, generalize=25)
