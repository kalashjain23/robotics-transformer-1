from torch.nn import Module
from torch import nn, Tensor
import torch


class TokenLearner(Module):
    def __init__(self):
        super().__init__()
        
        self.attention_mlp = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.GELU(),
            nn.Conv2d(512, 8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, spatial_tokens: Tensor):
        # spatial_tokens: B x C x H x W
        B, C, H, W = spatial_tokens.shape
        attention_maps = self.attention_mlp(spatial_tokens)
        tokens = torch.einsum('bchw,bkhw->bkc', spatial_tokens, attention_maps)
        return tokens / (H * W)
