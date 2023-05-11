"""
DNA Tokenizer template
- MetaSpace enforces preprocessing of substrings prior to sub-token detection.
https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb
- 

"""
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from pydantic import BaseModel
from typing import List, Tuple
from abc import ABC, abstractmethod
from tokenizers.trainers import BpeTrainer



class TokenizeRequest(BaseModel):
    sequence: str

class TokenizeResponse(BaseModel):
    tokens: List[int]

class AbstractTokenizer(ABC):
    @abstractmethod
    def train_from_generator(self, generator: List[str]):
        pass

    @abstractmethod
    def save(self, filename: str):
        pass

    @abstractmethod
    def load(self, filename: str):
        pass

    @abstractmethod
    def tokenize(self, sequence: str) -> List[str]:
        pass

    @abstractmethod
    def detokenize(self, tokens: List[str]) -> str:
        pass

    @abstractmethod
    def tokenize_request(self, request: TokenizeRequest) -> TokenizeResponse:
        pass

    @abstractmethod
    def detokenize_request(self, request: TokenizeRequest) -> TokenizeResponse:
        pass



class CustomTokenizer(AbstractTokenizer):
    def __init__(self, vocab_size: int = 20, min_frequency: int = 2):
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        self.tokenizer.decoder = decoders.Metaspace()
        
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[
                ("[CLS]", vocab_size + 1), 
                ("[SEP]", vocab_size + 2),
            ],
        )
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    def train_from_generator(self, generator: List[str]):
        self.tokenizer.train_from_iterator(generator, trainer=BpeTrainer(
        min_frequency=self.min_frequency,
        vocab_size=self.vocab_size
    ))

    def save(self, filename: str):
        self.tokenizer.save(filename)

    def load(self, filename: str):
        self.tokenizer = Tokenizer.from_file(filename)

    def tokenize(self, sequence: str) -> List[int]:
        encoded = self.tokenizer.encode(sequence)
        return encoded.ids

    def detokenize(self, tokens: List[int]) -> str:
        decoded = self.tokenizer.decode(tokens)
        return decoded

    def tokenize_request(self, request: TokenizeRequest) -> TokenizeResponse:
        tokens = self.tokenize(request.sequence)
        return TokenizeResponse(tokens=tokens)

    def detokenize_request(self, request: TokenizeRequest) -> TokenizeResponse:
        tokens = request.tokens
        detokenized = self.detokenize(tokens)
        return TokenizeRequest(sequence=detokenized)



def generate_sequence(file_path, sequence_length, max_sequences=None):
    with open(file_path, 'r') as file:
        buffer = ''
        num_sequences = 0
        for line in file:
            buffer += line.strip()
            while len(buffer) >= sequence_length:
                yield buffer[:sequence_length]
                buffer = buffer[sequence_length:]
                num_sequences += 1
                if max_sequences is not None and num_sequences >= max_sequences:
                    return




# Create an instance of the CustomTokenizer
tokenizer = CustomTokenizer()

# Train the tokenizer from a generator
# The generator needs to be sufficient.
generator = generate_sequence("NC_000002.12.txt", 1000, 500)
tokenizer.train_from_generator(generator)

# Save the tokenizer to a file
tokenizer.save("dna_tokenizer_20.json")

# Create a tokenization request using Pydantic models
request = TokenizeRequest(sequence="kenneth went to school")

# Tokenize the request
tokenized_request = tokenizer.tokenize_request(request)
print("Tokenized request:", tokenized_request.tokens)

# Detokenize the request
detokenized_request = tokenizer.detokenize_request(tokenized_request)
print("Detokenized request:", detokenized_request.sequence)