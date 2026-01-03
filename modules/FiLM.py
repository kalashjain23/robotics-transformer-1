from torch.nn import Module
from torch import nn, Tensor
import torch


class FiLM(Module):
    def __init__(self, language_dim: int, channels: int):
        super().__init__()
        
        self.channels = channels
        self.film_generator = nn.Sequential(
            nn.Linear(language_dim, language_dim),
            nn.ReLU(),
            nn.Linear(language_dim, 2 * self.channels)
        )
        
    def forward(self, visual_features: Tensor, language_embed: Tensor):
        film_params = self.film_generator(language_embed)
        
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        return (1 + gamma) * visual_features + beta