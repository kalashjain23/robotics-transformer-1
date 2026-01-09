from torch.nn import Module, Linear
import torch
from sentence_transformers import SentenceTransformer
from typing import List


class SentenceEncoder(Module):
    def __init__(self, output_dim: int = 512):
        super().__init__()
        
        self.output_dim = output_dim
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.projection = Linear(384, output_dim)
        
    def forward(self, text: List[str]):
        with torch.no_grad():
            embeddings = self.model.encode(text, convert_to_tensor=True)
            embeddings = embeddings.to(self.projection.weight.device)
            
        embeddings = embeddings.clone().detach()
        return self.projection(embeddings)