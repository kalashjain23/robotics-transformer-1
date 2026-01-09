# Robotics Transformer 1 (RT-1)

Implementation of RT-1, a Vision-Language-Action (VLA) model that takes video frames and natural language instructions as input and outputs robot actions.

For detailed module documentation, see [modules/README.md](modules/README.md).

## Installation

```bash
# Clone the repository
git clone git@github.com:kalashjain23/robotics-transformer-1.git
cd robotics-transformer-1

# Install dependencies using uv
uv sync
```

## Usage

```python
import torch
from modules.rt import RT

# Create the model
model = RT(
    d_model=512,           # Embedding dimension
    language_dim=512,      # Language embedding dimension
    heads=8,               # Attention heads
    num_images=6,          # Number of video frames
    tokens_per_image=8,    # Tokens per frame from TokenLearner
    d_ffn=2048,            # Feed-forward hidden dimension
    transformer_blocks=8,  # Number of transformer layers
)

# Prepare inputs
batch_size = 2
frames = torch.randn(batch_size, 6, 3, 300, 300)  # 6 RGB frames, 300x300
instructions = [
    "pick up the red apple from the table",
    "move the cup to the left drawer"
]

# Forward pass
actions = model(frames, instructions)
print(actions.shape)  # (2, 6, 11, 256)
```

## Output Format

The model outputs a tensor of shape `(B, num_images, num_actions, action_dim)`:
- **B**: Batch size
- **num_images**: 6 frames - one action prediction per frame
- **num_actions**: 11 action types (gripper, arm joints, base, etc.)
- **action_dim**: 256-dimensional embedding per action

## Running Tests

```bash
# Run all integration tests
uv run pytest tests/ -v

# Run with output
uv run pytest tests/ -v -s
```

## Project Structure

```
robotics-transformer-1/
├── modules/
│   ├── rt.py                  # Main RT model
│   ├── sentenceEncoder.py     # Language encoding
│   ├── efficientNetFiLM.py    # Vision backbone with FiLM
│   ├── FiLM.py                # Feature-wise Linear Modulation
│   ├── tokenLearner.py        # Token compression
│   ├── positionalEncoding.py  # Position information
│   ├── multiHeadAttention.py  # Self-attention
│   ├── transformerDecoder.py  # Transformer block
│   └── actionHead.py          # Action prediction head
├── tests/
│   └── test_integration.py    # Integration tests
├── pyproject.toml
└── README.md
```

## References

- [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)
- [Understanding The Attention Mechanism in Transformers with Code](https://medium.com/@lixue421/understanding-the-attention-mechanism-in-transformers-73ce20ead2ab) (helped me a lot!)
