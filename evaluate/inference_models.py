import torch
from typing import Literal


class EvalModel():
    def __init__(self, 
                 tokenizer, 
                 model, 
                 pooling, 
                 device):
        
        self.tokenizer = tokenizer
        self.model = model
        
        self.pooling = pooling
        self.pooling.to(device)
        
        self.model.to(device)
        self.device = device
        
        self.model.eval()
    
    def encode(self, x: str):
        with torch.no_grad():
            input_data = self.tokenizer.tokenize(x).to_torch()
            last_hidden_state = self.model(input_ids = input_data["input_ids"].to(self.device), 
                                           attention_mask = input_data["attention_mask"].to(self.device))
            y = self.pooling(last_hidden_state, attention_mask=input_data["attention_mask"].to(self.device))
            return torch.nn.functional.normalize(y.squeeze(), dim=0).detach().cpu().numpy()



class Baseline():
    def __init__(self, 
                 option: Literal["dna2bert", "nucleotide_transformer"], 
                 device: str = "cuda:1",
                 cache_dir: str = '/mnt/SSD2/pholur/dna2vec/baselines'):
        
        import os
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        self.device = device
        from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
        
        if option == "dna2bert":
            self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True)
            self.model = AutoModel.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True)
            self.model.to(device)
        
        elif option == "nucleotide_transformer":
            self.tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
            self.model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
            self.model.to(device)
            self.model.eval()
        
        else:
            raise NotImplementedError("Baseline is not implemented.")
        
        self.option = option
        
        
    def encode(self, x: list):
        with torch.no_grad():
            if self.option == "nucleotide_transformer":
                tokens_ids = self.tokenizer.batch_encode_plus(x, return_tensors="pt")["input_ids"]
                # Compute the embeddings
                attention_mask = tokens_ids != self.tokenizer.pad_token_id
                torch_outs = self.model(
                    tokens_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    encoder_attention_mask=attention_mask.to(self.device),
                    output_hidden_states=True
                )

                embeddings = torch_outs['hidden_states'][-1].detach().cpu().numpy()
                mean_sequence_embeddings = torch.sum(attention_mask.unsqueeze(-1)*embeddings, axis=-2)/torch.sum(attention_mask, axis=-1).unsqueeze(-1)
                return torch.nn.functional.normalize(mean_sequence_embeddings.squeeze(), dim=0).detach().cpu().numpy()
            
            elif self.option == "dna2bert":
                tokens_ids = self.tokenizer(x, return_tensors = 'pt')["input_ids"]
                attention_mask = tokens_ids != self.tokenizer.pad_token_id
                torch_outs = self.model(
                    tokens_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    encoder_attention_mask=attention_mask.to(self.device),
                    output_hidden_states=True
                )

                embeddings = torch_outs['hidden_states'][-1].detach().cpu().numpy()
                mean_sequence_embeddings = torch.sum(attention_mask.unsqueeze(-1)*embeddings, axis=-2)/torch.sum(attention_mask, axis=-1).unsqueeze(-1)
                return torch.nn.functional.normalize(mean_sequence_embeddings.squeeze(), dim=0).detach().cpu().numpy()
            
            else:
                raise NotImplementedError("Baseline Option undefined")


    def get_sentence_embedding_dimension(self):
        if self.option == "dna2bert":
            return 768
        
        elif self.option == "nucleotide_transformer":
            return 2560