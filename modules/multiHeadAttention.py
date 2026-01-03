from torch import Tensor
from torch import nn
from torch.nn import Module
import torch.nn.functional as F


class MultiHeadAttention(Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
    def forward(self, x: Tensor):
        batch_size, seq_len, _ = x.shape
        
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        
        Q = Q.view(batch_size, seq_len, self.heads, self.d_k)
        K = K.view(batch_size, seq_len, self.heads, self.d_k)
        V = V.view(batch_size, seq_len, self.heads, self.d_k)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        output, attention_weights = self._scaledDotProductAttention(Q, K, V)
        output = output.transpose(1, 2)
        output = output.contiguous().view(batch_size, seq_len, self.d_model)
        output = self.linear_out(output)
        
        return output
        
    def _scaledDotProductAttention(self, query: Tensor, key: Tensor, value: Tensor):
        attention_scores = query @ key.transpose(-2, -1)
        attention_weights = F.softmax(attention_scores / (self.d_k**0.5), dim=-1)
        return attention_weights @ value, attention_weights
        
