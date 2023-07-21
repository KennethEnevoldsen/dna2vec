"""
Pinecone store for DNA2Vec embeddings

This usually crashes in the first run but succeeds in the second 0_o
Issues with API timeout on DB creation
Unstable: https://github.com/pinecone-io/pinecone-python-client/issues
"""
import string
import pinecone
import torch
from tqdm import tqdm
import random



class EvalModel():
    def __init__(self, tokenizer, model, pooling, device):
        self.tokenizer = tokenizer
        self.model = model
        
        self.pooling = pooling
        self.pooling.to(device)
        
        self.model.to(device)
        self.device = device
        
        self.model.eval()
    
    def encode(self, x):
        with torch.no_grad():
            input_data = self.tokenizer.tokenize(x).to_torch()
            last_hidden_state = self.model(input_ids = input_data["input_ids"].to(self.device), 
                                           attention_mask = input_data["attention_mask"].to(self.device))
            y = self.pooling(last_hidden_state, attention_mask=input_data["attention_mask"].to(self.device))
            return torch.nn.functional.normalize(y.squeeze(), dim=0).detach().cpu().numpy()

        

class PineconeStore:
    def __init__(
        self,
        device,
        index_name: str,
        metric: str = "cosine",
        model_params = None,
    ):
        if model_params == None:
            raise ValueError("Model params are empty.")
        else:
            self.model = EvalModel(
                model_params["tokenizer"],
                model_params["model"],
                model_params["pooling"],
                device = device
            )
        
        if "config-" in index_name: # premium account
            self.api_key = "ded0a046-d0fe-4f8a-b45c-1d6274ad555e"
            self.environment = "us-west4-gcp"
            
        else:
            raise NotImplementedError("Name not identified.")
        
        self.initialize_pinecone_upsertion(metric, index_name)
        self.index_name = index_name

    def initialize_pinecone_upsertion(
        self, 
        metric: str, 
        index_name: str
    ):
        pinecone.init(
            api_key=self.api_key, 
            environment=self.environment
        )

        # only create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            print(f"Creating new index, {index_name}")
            
            try:
                dimension = self.model.get_sentence_embedding_dimension()
            except:
                dimension = 384
                
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric
            )

        # now connect to the index
        self.index = pinecone.GRPCIndex(index_name)

    @staticmethod
    def batched_data_generator(file_path, batch_size):
        """Generator to stream many snippets from text file

        Args:
            file_path (str): Location of file to stream individual snippets
            batch_size (int): stream load

        Yields:
            list(str): list of strings
        """
        import pickle
        
        with open(file_path, "rb") as f:
            list_of_objects = pickle.load(f)
            batch = []
            for unit in list_of_objects:
                batch.append(unit)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    @staticmethod
    def generate_random_string(length: str = 20):
        letters = string.ascii_lowercase
        return "".join(random.choice(letters) for _ in range(length))

    def trigger_pinecone_upsertion(self, file_paths: list, 
                                   batch_size: int = 64):
        from tqdm import tqdm
        
        for file_path in file_paths:
            batches = PineconeStore.batched_data_generator(file_path, batch_size)

            for batch in tqdm(batches):
                ids = [PineconeStore.generate_random_string() for _ in range(len(batch))]

                # create metadata batch - we can add context here
                metadatas = batch
                texts = [text["text"] for text in batch]
                # create embeddings
                xc = self.model.encode(texts)

                # create records list for upsert
                records = zip(ids, xc, metadatas)
                # upsert to Pinecone
                self.index.upsert(vectors=records)

        # check number of records in the index
        self.index.describe_index_stats()

    def query(self, query, top_k=5):  # consider batching if slow
        # create the query vector
        xq = self.model.encode(query).tolist()

        # now query
        xc = self.index.query(xq, top_k=top_k, include_metadata=True)

        return xc

    def drop_table(self):  # times out for large data!
        pinecone.delete_index(self.index_name)






if __name__ == "__main__":
    import argparse
    import random

    # fmt: off
    parser = argparse.ArgumentParser(description="Pinecone parse and upload.")
    parser.add_argument('--reupload', help="Should we reupload the data?", type=str, choices=['y','n'])
    parser.add_argument('--drop', help="Should we drop the data?", type=str, choices=['y','n'])
    parser.add_argument('--inputpath', help="Splice input file", type=str)
    parser.add_argument('--indexname', help="Index name", type=str)
    
    args = parser.parse_args()
    # fmt: on

    random.seed(42)

    pinecone_obj = PineconeStore(
        device="cuda:4", 
        index_name=args.indexname
    )

    # if args.reupload == "y":
    #     pinecone_obj.trigger_pinecone_upsertion(
    #         file_path=args.inputpath
    #     )

    if args.drop == "y":
        pinecone_obj.drop_table()
