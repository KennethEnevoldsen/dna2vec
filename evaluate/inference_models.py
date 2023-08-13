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
                 option: Literal["dna2bert-max", "dna2bert-mean", "nucleotide-transformer"], 
                 device: str = "cuda:1",
                 cache_dir: str = '/mnt/SSD2/pholur/dna2vec/baselines'):
        
        import os
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        self.device = device
        from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
        
        if option == "dna2bert-mean" or option == "dna2bert-max":
            self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
            self.model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
            self.model.to(device)
            self.model.eval()

        elif option == "nucleotide-transformer":
            self.tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
            self.model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
            self.model.to(device)
            self.model.eval()
        
        else:
            raise NotImplementedError("Baseline is not implemented.")
        
        self.option = option
        
        
    def encode(self, x: list):
        with torch.no_grad():
            if self.option == "nucleotide-transformer":
                tokens_ids = self.tokenizer.batch_encode_plus(x, return_tensors="pt", padding=True, truncation=True)["input_ids"]
                # Compute the embeddings
                attention_mask = tokens_ids != self.tokenizer.pad_token_id
                torch_outs = self.model(
                    tokens_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    encoder_attention_mask=attention_mask.to(self.device),
                    output_hidden_states=True
                )
                
                embeddings = torch_outs['hidden_states'][-1]
                mean_sequence_embeddings = torch.sum(
                    attention_mask.to(self.device).unsqueeze(-1)*embeddings, axis=-2)/\
                        torch.sum(attention_mask.to(self.device), axis=-1).unsqueeze(-1)
                return torch.nn.functional.normalize(mean_sequence_embeddings.squeeze(), dim=0).detach().cpu().numpy()
            
            elif self.option == "dna2bert-mean":
                tokens_ids = self.tokenizer(x, return_tensors = 'pt', padding=True)["input_ids"]
                attention_mask = tokens_ids != self.tokenizer.pad_token_id
                
                torch_outs = self.model(
                    tokens_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    encoder_attention_mask=attention_mask.to(self.device),
                    output_hidden_states=True,
                )

                embeddings = torch_outs[0]
                mean_sequence_embeddings = torch.sum(
                    attention_mask.to(self.device).unsqueeze(-1)*embeddings, axis=-2)/\
                        torch.sum(attention_mask.to(self.device), axis=-1).unsqueeze(-1)
                return torch.nn.functional.normalize(mean_sequence_embeddings.squeeze(), dim=0).detach().cpu().numpy()
            
            elif self.option == "dna2bert-max":
                tokens_ids = self.tokenizer(x, return_tensors = 'pt', padding=True)["input_ids"]
                attention_mask = tokens_ids != self.tokenizer.pad_token_id
                
                torch_outs = self.model(
                    tokens_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    encoder_attention_mask=attention_mask.to(self.device),
                    output_hidden_states=True,
                )

                embeddings = torch_outs[0]
                max_sequence_embeddings = torch.max(
                    attention_mask.to(self.device).unsqueeze(-1)*embeddings, axis=-2)
                return torch.nn.functional.normalize(max_sequence_embeddings.values.squeeze(), dim=0).detach().cpu().numpy()
            else:
                raise NotImplementedError("Baseline Option undefined")


    def get_sentence_embedding_dimension(self):
        if self.option == "dna2bert-max" or self.option == "dna2bert-mean":
            return 768
        
        elif self.option == "nucleotide-transformer":
            return 1280