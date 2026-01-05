from torch import Module, nn, Tensor
from multiHeadAttention import MultiHeadAttention


class TransformerDecoder(Module):
    def __init__(self, d_model, heads, d_feed_forward):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, heads)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_feed_forward),
            nn.GELU(),
            nn.Linear(d_model, d_feed_forward),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: Tensor):
        attn_out = self.attention(x)
        x = self.norm(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x