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
            last_hidden_state = self.model(**input_data)
            y = self.pooling(last_hidden_state, attention_mask=input_data["attention_mask"])
            return y.squeeze().detach().cpu().numpy()

        

class PineconeStore:
    def __init__(
        self,
        device: str,
        index_name: str = "dna-1-0504",
        metric: str = "cosine",
        model_params = None,
    ):
        if model_params == None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        else:
            self.model = EvalModel(
                model_params["tokenizer"],
                model_params["model"],
                model_params["pooling"],
                device = device
            )
        
        self.initialize_pinecone_upsertion(metric, index_name)
        self.index_name = index_name

    def initialize_pinecone_upsertion(
        self, metric: str = "cosine", index_name: str = "dna-1-0504"
    ):
        pinecone.init(
            api_key="ded0a046-d0fe-4f8a-b45c-1d6274ad555e", environment="us-west4-gcp"
        )

        # only create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
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
        with open(file_path, "r", encoding="utf-8") as f:
            batch = []
            for line in f:
                batch.append(line.strip())
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    @staticmethod
    def generate_random_string(length: str = 20):
        letters = string.ascii_lowercase
        return "".join(random.choice(letters) for _ in range(length))

    def trigger_pinecone_upsertion(self, file_path: str, batch_size: int = 64):
        from tqdm import tqdm

        batches = PineconeStore.batched_data_generator(file_path, batch_size)

        for batch in tqdm(batches):
            ids = [PineconeStore.generate_random_string() for _ in range(len(batch))]

            # create metadata batch - we can add context here
            metadatas = [{"text": text} for text in batch]
            # create embeddings
            xc = self.model.encode(batch)

            # create records list for upsert
            records = zip(ids, xc, metadatas)
            # upsert to Pinecone
            self.index.upsert(vectors=records)

        # check number of records in the index
        self.index.describe_index_stats()

    def query(self, query):  # consider batching if slow
        # create the query vector
        xq = self.model.encode(query).tolist()

        # now query
        xc = self.index.query(xq, top_k=3, include_metadata=True)

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
    args = parser.parse_args()
    # fmt: on

    random.seed(42)

    pinecone_obj = PineconeStore(device="cuda:4")

    if args.reupload == "y":
        pinecone_obj.trigger_pinecone_upsertion(file_path=args.inputpath)

    for _ in tqdm(range(10000)):  # robustness check
        pinecone_obj.query("OMEGALUL IT WORKS 0_o")

    if args.drop == "y":
        pinecone_obj.drop_table()
