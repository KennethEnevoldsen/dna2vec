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




def main(paths:list,
         store: PineconeStore = None,
         config: str = None,
         test_k: int = 200,
         top_k: int = 50,
         generalize: int = 5,
         edit_mode: Literal["random deplete", "end deplete", "swap"] = "end deplete"
):
    per_samples = test_k // len(paths)
    test_lines = []
    print(paths)
    for path in paths:
        test_lines.extend(pick_random_lines(path, per_samples))
    train_flags = np.zeros((test_k, len(test_lines[0]["text"])))   
     
    for i,line in tqdm(enumerate(test_lines)):
        
        query = line["text"]
        index = line["position"]
        metadata = line["metadata"]
        
        train_flag = permute(store, query, index, top_k,  metadata, edit_mode, generalize)
        train_flags[i,:] = train_flag
        
    np.savez_compressed(f"test_cache/permute/refactored_permute_\
{config}_{edit_mode}_{test_k}_{top_k}_{generalize}.npz", train = train_flags)
        



if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    
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
                
        main(list_of_data_sources, store, config, top_k=5, edit_mode=args.mode, test_k = 500, generalize=20)
        main(list_of_data_sources, store, config, top_k=50, edit_mode=args.mode, test_k = 500, generalize=20)
    pass
