import sys
sys.path.append("../src/")
import torch

from tests.pinecone_store import PineconeStore


checkpoints = {}

checkpoints["trained"] = {}
checkpoints["trained"]["model"] = torch.load("/mnt/SSD5/pholur/checkpoints/checkpoint_smooth-rock-29.pt")

checkpoints["init"] = {}
checkpoints["init"]["model"] = torch.load("/mnt/SSD5/pholur/checkpoints/checkpoint_initalialized.pt")

print("Checkpoints loaded.")

from dna2vec.trainer import ContrastiveTrainer
from dna2vec.model import model_from_config
# The path for the tokenizer isn't relative.


for baseline in checkpoints:
    config = checkpoints[baseline]["model"]["config"]
    config.model_config.tokenizer_path = "/home/pholur/dna2vec/src/model/tokenizers/dna_tokenizer_10k.json"
    encoder, pooling, tokenizer = model_from_config(config.model_config)

    encoder.load_state_dict(checkpoints[baseline]["model"]["model"])
    encoder.eval()
    
    checkpoints[baseline]["encoder"] = encoder
    checkpoints[baseline]["pooling"] = pooling
    checkpoints[baseline]["tokenizer"] = tokenizer
    checkpoints[baseline]["pineconestore"] = PineconeStore(
                                                                device = torch.device("cuda:3"),
                                                                index_name = baseline,
                                                                metric = "cosine",
                                                                model_params = {
                                                                    "tokenizer": tokenizer,
                                                                    "model": encoder,
                                                                    "pooling": pooling
                                                                }
                                                            )


# checkpoints["init"]["pineconestore"].drop_table()
# checkpoints["trained"]["pineconestore"].drop_table()


# checkpoints["init"]["pineconestore"].trigger_pinecone_upsertion("/home/pholur/dna2vec/tests/data/subsequences_sample_train.txt")
# checkpoints["trained"]["pineconestore"].trigger_pinecone_upsertion("/home/pholur/dna2vec/tests/data/subsequences_sample_train.txt")
checkpoints["trained"]["pineconestore"].trigger_pinecone_upsertion("/home/pholur/dna2vec/tests/data/subsequences_sample_train_chromosome3.txt")