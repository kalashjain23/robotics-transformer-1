from torch.nn import Module, ModuleList
from torch import Tensor

from modules.actionHead import ActionHead
from modules.efficientNetFiLM import EfficientNetFiLM
from modules.positionalEncoding import PositionalEncoding
from modules.tokenLearner import TokenLearner
from modules.transformerDecoder import TransformerDecoder
from modules.sentenceEncoder import SentenceEncoder


class RT(Module):
    def __init__(
        self,
        d_model: int = 512,
        language_dim: int = 512,
        heads: int = 8,
        num_images:int = 6,
        tokens_per_image:int = 8,
        d_ffn:int= 2048,
        transformer_blocks:int = 8,
    ):
        super().__init__()
        self.tokens_per_image = tokens_per_image
        
        self.sentence_encoder = SentenceEncoder(output_dim=d_model)
        
        self.efficient_net_film = EfficientNetFiLM(language_dim=language_dim, output_dim=d_model)
        
        self.token_learner = TokenLearner()
        
        self.positional_encoding = PositionalEncoding(d_model=d_model, seq_len=num_images * tokens_per_image)
        
        self.transformer_decoder = ModuleList([
            TransformerDecoder(
                d_model=d_model,
                heads=heads,
                d_feed_forward=d_ffn
            ) for _ in range(transformer_blocks)
        ])
        
        self.action_head = ActionHead()
        
    def forward(self, frames: Tensor, instructions):
        B, num_images, C, H, W = frames.shape
        
        language_embed = self.sentence_encoder(instructions)
        language_embed_expanded = language_embed.unsqueeze(1).repeat(1, num_images, 1)
        language_embed_flat = language_embed_expanded.reshape(B * num_images, -1)
        
        frames = frames.reshape(B * num_images, C, H, W)
        
        vision_features = self.efficient_net_film(frames, language_embed_flat)
        
        vision_tokens = self.token_learner(vision_features)
        
        vision_tokens = vision_tokens.reshape(B, num_images * self.tokens_per_image, -1)
        
        vision_tokens = self.positional_encoding(vision_tokens)
        
        x = vision_tokens
        for block in self.transformer_decoder:
            x = block(x)
            
        return self.action_head(x)
