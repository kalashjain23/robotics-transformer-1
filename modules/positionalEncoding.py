from torch import Tensor
from torch.nn import Module
import torch
import math


class PositionalEncoding(Module):
    def __init__(self, d_model: int, seq_len: int):
        super().__init__()
        
        self.d_model = d_model
        self.pe = torch.zeros(seq_len, d_model)
        self.position = torch.arange(0, seq_len).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            (-math.log(10000.0) / d_model)
        )
        
        self.pe[:, 0::2] = torch.sin(self.position * div_term)
        self.pe[:, 1::2] = torch.cos(self.position * div_term)
        
        self.register_buffer('pe', self.pe)
        
    def forward(self, x: Tensor):
        return x + self.pe[:x.size(1), :]