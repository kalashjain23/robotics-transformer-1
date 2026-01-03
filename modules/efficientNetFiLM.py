from torch.nn import Module
from torch import nn, Tensor
from torchvision.models import efficientnet_b3
from FiLM import FiLM


class EfficientNetFiLM(Module):
    def __init__(self, language_dim: int, output_dim: int = 512):
        super().__init__()
        
        self.efficient_net = efficientnet_b3(pretrained=True)
        self.features = self.efficient_net.features
            
        self.mbconv_channels = [
            24, 24,                           # Stage 1 (2 blocks)
            32, 32, 32,                       # Stage 2 (3 blocks)
            48, 48, 48,                       # Stage 3 (3 blocks)
            96, 96, 96, 96, 96,               # Stage 4 (5 blocks)
            136, 136, 136, 136, 136,          # Stage 5 (5 blocks)
            232, 232, 232, 232, 232, 232,     # Stage 6 (6 blocks)
            384, 384                          # Stage 7 (2 blocks)
        ]
        
        self.film_layers = nn.ModuleList([
            FiLM(language_dim, channel)
            for channel in self.mbconv_channels
        ])
        
        self.projection = nn.Conv2d(384, output_dim, kernel_size=1)
        
    def forward(self, image: Tensor, language_embed: Tensor):
        x: Tensor = image
        film_idx = 0
        
        # inital conv layer
        x = self.features[0](x)
        
        for stage_idx in range(1, 8):
            stage = self.features[stage_idx]
            if isinstance(stage, nn.Sequential):
                for block in stage:
                    x = block(x)
                    x = self.film_layers[film_idx](x, language_embed)
                    film_idx += 1
                
        # final conv layer
        x = self.projection(x)

        x = x.flatten(2).transpose(1, 2)
        
        return x
        