from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch
from dna2vec.model import Encoder, AveragePooler, BPTokenizer
import random
import numpy as np

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


def load_local_model():
    model_path = "/mnt/SSD1/yigit/models/regularization_0.0_pool_1020mean_checkpoint.pt"
    tokenizer_path = "/mnt/SSD1/yigit/dna_tokenizer_10k.json"

    # Load the model weights
    info_dict = torch.load(model_path)
    model_kwargs = info_dict['config'].model_config.model_dump()
    model_kwargs = {k: v for k, v in model_kwargs.items() if k not in ["tokenizer_path", "pooling","pos_embedding"]}

    model = Encoder(**model_kwargs)
    model.load_state_dict(info_dict["model"])
    model.eval()
    tokenizer = BPTokenizer(vocab_size=model_kwargs["vocab_size"])
    tokenizer = tokenizer.load(tokenizer_path)
    pooler = AveragePooler()

    return model, tokenizer, pooler

def generate_dna_sequence(length=200):
    return ''.join(random.choice('ATGC') for _ in range(length))


def compare_models():
    hf_model, hf_tokenizer, _ = load_hf_model()
    local_model, local_tokenizer, _ = load_local_model()
    
    for i in range(10):
        dummy_input = generate_dna_sequence(200)
        
        tokenized_input_hf = hf_tokenizer(dummy_input, return_tensors="pt")
        tokenized_input_local = local_tokenizer.tokenize([dummy_input]).encodings[0]
        
        # compare the tokenized inputs as numpy arrays
        np_tokenized_input_hf_ids = tokenized_input_hf.input_ids.detach().numpy()
        np_tokenized_input_local_ids = np.array(tokenized_input_local.ids).reshape(1, -1)
        
        print("tokenized input difference: ", np.sum(np.abs(np_tokenized_input_hf_ids - np_tokenized_input_local_ids)))

        # compare the outputs
        output_hf = hf_model(**tokenized_input_hf)
        output_local = local_model.forward(torch.tensor(tokenized_input_local.ids).reshape(1, -1), torch.tensor(tokenized_input_local.attention_mask).reshape(1, -1))
        
        # compare the outputs as numpy arrays
        np_output_hf = output_hf.detach().numpy()
        np_output_local = output_local.detach().numpy()
        
        print("output difference: ", np.sum(np.abs(np_output_hf - np_output_local)))
        
        print("________________________________________________________")
        
        
        
if __name__ == "__main__":
    compare_models()
        
        
    





