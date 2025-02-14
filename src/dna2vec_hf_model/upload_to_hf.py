from dna2vec_hf_model.configuration_dna2vec import DNAEncoderConfig
from dna2vec_hf_model.modeling_dna2vec import DNAEncoder
import torch
from huggingface_hub import login
import math
from dna2vec.tokenizer import BPTokenizer
from transformers import PreTrainedTokenizerFast
DNAEncoderConfig.register_for_auto_class()
DNAEncoder.register_for_auto_class("AutoModel")

model_path = "/mnt/SSD1/yigit/models/regularization_0.0_pool_1020mean_checkpoint.pt"
tokenizer_path = "/mnt/SSD1/yigit/dna_tokenizer_10k.json"

# Load the model weights
info_dict = torch.load(model_path)
model_kwargs = info_dict['config'].model_config.model_dump()
model_kwargs = {k: v for k, v in model_kwargs.items() if k not in ["tokenizer_path", "pooling","pos_embedding"]}

# Create config and model
config = DNAEncoderConfig(**model_kwargs)
model = DNAEncoder(config)
# delete a key from info_dict["model"]

position = torch.arange(config.max_position_embeddings).unsqueeze(1)
div_term = torch.exp(
    torch.arange(0, config.embedding_dim, 2) * (-math.log(10000.0) / config.embedding_dim)
)
pe = torch.zeros(config.max_position_embeddings, 1, config.embedding_dim)
pe[:, 0, 0::2] = torch.sin(position * div_term)
pe[:, 0, 1::2] = torch.cos(position * div_term)
pe = pe.squeeze(1).unsqueeze(0)

del info_dict["model"]["positional_embedding.pe"]
info_dict["model"]["positional_embedding"] = pe

model.encoder.load_state_dict(info_dict["model"])
tokenizer = BPTokenizer(vocab_size=config.vocab_size)
tokenizer = tokenizer.load(tokenizer_path)
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

# Login to Hugging Face
login()

# Push to hub
repo_name = "roychowdhuryresearch/dna2vec"
model.push_to_hub(repo_name, pipeline_tag="sentence-similarity")
tokenizer.push_to_hub(repo_name)
