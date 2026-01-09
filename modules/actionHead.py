from torch import nn, Tensor
from torch.nn import Module


class ActionHead(Module):
    def __init__(
        self, 
        input_dim: int = 512, 
        output_dim: int = 256, 
        num_actions: int = 11,
        num_images: int = 6,
        tokens_per_image: int = 8
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_actions = num_actions
        self.num_images = num_images
        self.tokens_per_image = tokens_per_image
        
        self.head = nn.Linear(input_dim, num_actions * output_dim)
        
    def forward(self, x: Tensor):
        B = x.shape[0]
        x = x.reshape(B, self.num_images, self.tokens_per_image, self.input_dim)
        
        # last token of each image
        x = x[:, :, -1, :]
        
        logits = self.head(x)
        logits = logits.reshape(B, self.num_images, self.num_actions, self.output_dim)
        
        return logits