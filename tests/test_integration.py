"""
Integration tests for the RT-1 (Robotics Transformer) model.

These tests verify that all components work together correctly
and that the full forward pass produces expected outputs.
"""

import pytest
import torch

from modules.rt import RT
from modules.sentenceEncoder import SentenceEncoder
from modules.efficientNetFiLM import EfficientNetFiLM
from modules.FiLM import FiLM
from modules.tokenLearner import TokenLearner
from modules.positionalEncoding import PositionalEncoding
from modules.multiHeadAttention import MultiHeadAttention
from modules.transformerDecoder import TransformerDecoder
from modules.actionHead import ActionHead


class TestFullPipeline:
    """Test the complete RT-1 pipeline from input to output."""

    @pytest.fixture
    def model(self):
        """Create a standard RT model for testing."""
        return RT(
            d_model=512,
            language_dim=512,
            heads=8,
            num_images=6,
            tokens_per_image=8,
            d_ffn=2048,
            transformer_blocks=8,
        )

    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for testing."""
        batch_size = 2
        frames = torch.randn(batch_size, 6, 3, 300, 300)
        instructions = [
            "pick up the red apple from the table",
            "move the cup to the left drawer"
        ]
        return frames, instructions

    def test_model_instantiation(self, model):
        """Test that the model can be created with default parameters."""
        assert model is not None
        assert hasattr(model, 'sentence_encoder')
        assert hasattr(model, 'efficient_net_film')
        assert hasattr(model, 'token_learner')
        assert hasattr(model, 'positional_encoding')
        assert hasattr(model, 'transformer_decoder')
        assert hasattr(model, 'action_head')

    def test_forward_pass_output_shape(self, model, sample_inputs):
        """Test that forward pass produces correct output shape."""
        frames, instructions = sample_inputs
        batch_size = frames.shape[0]

        actions = model(frames, instructions)

        assert actions.shape == (batch_size, 6, 11, 256), \
            f"Expected (2, 6, 11, 256), got {actions.shape}"

    def test_forward_pass_output_dtype(self, model, sample_inputs):
        """Test that output tensor has correct dtype."""
        frames, instructions = sample_inputs
        actions = model(frames, instructions)

        assert actions.dtype == torch.float32

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_variable_batch_sizes(self, model, batch_size):
        """Test that the model handles different batch sizes."""
        frames = torch.randn(batch_size, 6, 3, 300, 300)
        instructions = ["test instruction"] * batch_size

        actions = model(frames, instructions)

        assert actions.shape == (batch_size, 6, 11, 256)

    def test_gradient_flow(self, model, sample_inputs):
        """Test that gradients flow through the entire model."""
        frames, instructions = sample_inputs
        frames.requires_grad = True

        model.train()
        actions = model(frames, instructions)
        loss = actions.sum()
        loss.backward()

        assert frames.grad is not None, "Gradients should flow to input"
        assert not torch.isnan(frames.grad).any(), "Gradients should not be NaN"

    def test_model_eval_mode(self, model, sample_inputs):
        """Test that model works in eval mode."""
        frames, instructions = sample_inputs

        model.eval()
        with torch.no_grad():
            actions = model(frames, instructions)

        assert actions.shape == (2, 6, 11, 256)


class TestVisionLanguageIntegration:
    """Test the integration of vision and language components."""

    def test_language_conditions_vision(self):
        """Test that language embeddings condition visual features via FiLM."""
        batch_size = 2
        language_dim = 512
        channels = 64

        film = FiLM(language_dim=language_dim, channels=channels)
        visual_features = torch.randn(batch_size, channels, 10, 10)
        lang_embed1 = torch.randn(batch_size, language_dim)
        lang_embed2 = torch.randn(batch_size, language_dim)

        out1 = film(visual_features, lang_embed1)
        out2 = film(visual_features, lang_embed2)

        # Different language embeddings should produce different outputs
        assert not torch.allclose(out1, out2), \
            "Different language should produce different visual modulation"

    def test_sentence_encoder_projection(self):
        """Test that sentence encoder projects to correct dimension."""
        encoder = SentenceEncoder(output_dim=512)
        texts = ["pick up the apple", "move the cup"]

        embeddings = encoder(texts)

        assert embeddings.shape == (2, 512)

    def test_efficientnet_film_output(self):
        """Test EfficientNetFiLM produces spatial features."""
        model = EfficientNetFiLM(language_dim=512, output_dim=512)
        images = torch.randn(2, 3, 300, 300)
        lang_embed = torch.randn(2, 512)

        features = model(images, lang_embed)

        # Should produce spatial feature maps
        assert len(features.shape) == 4  # B, C, H, W
        assert features.shape[1] == 512  # output_dim channels


class TestTokenProcessing:
    """Test token learning and positional encoding integration."""

    def test_token_learner_compression(self):
        """Test that TokenLearner compresses spatial features to tokens."""
        token_learner = TokenLearner()
        spatial_features = torch.randn(2, 512, 10, 10)  # B, C, H, W

        tokens = token_learner(spatial_features)

        assert tokens.shape == (2, 8, 512), \
            f"Expected (2, 8, 512), got {tokens.shape}"

    def test_positional_encoding_addition(self):
        """Test that positional encoding adds position info without changing shape."""
        seq_len = 48  # 6 images * 8 tokens
        d_model = 512
        pos_enc = PositionalEncoding(d_model=d_model, seq_len=seq_len)

        tokens = torch.randn(2, seq_len, d_model)
        encoded = pos_enc(tokens)

        assert encoded.shape == tokens.shape
        # Positional encoding should change the values
        assert not torch.allclose(encoded, tokens)


class TestTransformerProcessing:
    """Test transformer decoder integration."""

    def test_attention_preserves_shape(self):
        """Test that multi-head attention preserves sequence shape."""
        attention = MultiHeadAttention(d_model=512, heads=8)
        x = torch.randn(2, 48, 512)

        output = attention(x)

        assert output.shape == x.shape

    def test_decoder_block_integration(self):
        """Test that decoder block processes tokens correctly."""
        decoder = TransformerDecoder(d_model=512, heads=8, d_feed_forward=2048)
        x = torch.randn(2, 48, 512)

        output = decoder(x)

        assert output.shape == x.shape
        # Output should be different from input (not identity)
        assert not torch.allclose(output, x)

    def test_stacked_decoders(self):
        """Test that multiple decoder blocks can be stacked."""
        decoders = torch.nn.ModuleList([
            TransformerDecoder(d_model=512, heads=8, d_feed_forward=2048)
            for _ in range(8)
        ])
        x = torch.randn(2, 48, 512)

        for decoder in decoders:
            x = decoder(x)

        assert x.shape == (2, 48, 512)


class TestActionPrediction:
    """Test action head and final prediction."""

    def test_action_head_output_shape(self):
        """Test that action head produces correct output shape."""
        action_head = ActionHead(
            input_dim=512,
            output_dim=256,
            num_actions=11,
            num_images=6,
            tokens_per_image=8
        )
        x = torch.randn(2, 48, 512)  # B, seq_len, d_model

        actions = action_head(x)

        assert actions.shape == (2, 6, 11, 256)

    def test_action_head_uses_last_token(self):
        """Test that action head uses the last token per image."""
        action_head = ActionHead(
            input_dim=512,
            output_dim=256,
            num_actions=11,
            num_images=6,
            tokens_per_image=8
        )

        # Create input where only last tokens (indices 7, 15, 23, 31, 39, 47) matter
        x = torch.zeros(1, 48, 512)

        # Set distinctive values at last token positions
        for i in range(6):
            last_token_idx = (i + 1) * 8 - 1
            x[0, last_token_idx, :] = float(i + 1)

        actions = action_head(x)

        # Each image's action should be based on its last token
        # (won't be exactly equal due to linear projection, but should be different)
        for i in range(5):
            assert not torch.allclose(actions[0, i], actions[0, i + 1])


class TestEndToEndScenarios:
    """Test realistic usage scenarios."""

    def test_single_instruction_batch(self):
        """Test processing a single instruction."""
        model = RT()
        frames = torch.randn(1, 6, 3, 300, 300)
        instructions = ["grasp the blue cube"]

        actions = model(frames, instructions)

        assert actions.shape == (1, 6, 11, 256)

    def test_different_instructions_same_frames(self):
        """Test that different instructions produce different actions for same frames."""
        model = RT()
        model.eval()

        frames = torch.randn(1, 6, 3, 300, 300)

        with torch.no_grad():
            actions1 = model(frames, ["pick up the red apple"])
            actions2 = model(frames, ["move to the left"])

        # Different instructions should lead to different action predictions
        assert not torch.allclose(actions1, actions2), \
            "Different instructions should produce different actions"

    def test_parameter_count(self):
        """Test that model has expected number of parameters."""
        model = RT()
        total_params = sum(p.numel() for p in model.parameters())

        # Model should have substantial parameters (rough check)
        assert total_params > 10_000_000, \
            f"Model seems too small: {total_params:,} parameters"

    def test_deterministic_output(self):
        """Test that same inputs produce same outputs in eval mode."""
        model = RT()
        model.eval()

        frames = torch.randn(1, 6, 3, 300, 300)
        instructions = ["test instruction"]

        with torch.no_grad():
            actions1 = model(frames, instructions)
            actions2 = model(frames, instructions)

        assert torch.allclose(actions1, actions2), \
            "Same inputs should produce same outputs in eval mode"
