# 불확실성 추정기 구현 예정 

import torch
import torch.nn as nn
import torch.nn.functional as F

# Robust import for config and transformer_blocks
try:
    from .. import config
    from .transformer_blocks import TransformerEncoderLayer, PositionalEncoding # PositionalEncoding might not be directly used if input is global feature
except ImportError:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir) # src/
    project_root = os.path.dirname(parent_dir) # UAIC_NEW/
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    import config
    from models.transformer_blocks import TransformerEncoderLayer, PositionalEncoding

class UncertaintyEstimator(nn.Module):
    def __init__(self, vocab_size: int, 
                 image_feature_dim: int = config.IMAGE_EMBED_SIZE, 
                 d_model: int = config.HIDDEN_SIZE, 
                 num_layers: int = config.UE_NUM_LAYERS, 
                 num_heads: int = config.NUM_ATTENTION_HEADS, 
                 d_ff: int = config.FEED_FORWARD_SIZE, 
                 dropout: float = config.DROPOUT_PROB):
        super(UncertaintyEstimator, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # Input projection with batch normalization
        self.input_proj = nn.Sequential(
            nn.Linear(image_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # The paper mentions "Transformer self-attention". 
        # If the input is a global image feature (batch_size, feature_dim),
        # it needs to be treated as a sequence of length 1 for a standard TransformerEncoder.
        # Or, the self-attention is applied over a sequence of regional features if the ImageEncoder provides them.
        # Assuming global features for now, which are expanded to seq_len=1.
        # Positional encoding might be trivial for seq_len=1 but included for completeness if features were sequential.
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=10) # Max_len=10 is arbitrary for seq_len=1

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # MLP head with batch normalization
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, self.vocab_size)
        )
        
        # The paper states pi is in [-1, 0]. We can achieve this by scaling tanh output.
        # tanh(x) is in [-1, 1]. (tanh(x) - 1) / 2 is in [-1, 0].
        # Or, directly use a linear layer and rely on the MSE loss with target in [-1,0]
        # Forcing it with activation might be more stable. Let's use (tanh(x)-1)/2.

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features (torch.Tensor): Global image features from ImageEncoder, 
                                         shape (batch_size, image_feature_dim).
        Returns:
            torch.Tensor: Uncertainty scores (pi) for each word in the vocabulary,
                          shape (batch_size, vocab_size), values in [-1, 0].
        """
        # Project and normalize input features
        x = self.input_proj(image_features)
        
        # Add sequence dimension
        x = x.unsqueeze(1)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        for layer in self.encoder_layers:
            x = layer(x, src_mask=None)
        
        x = x.squeeze(1)
        
        # MLP head with gradual dimension reduction
        pi_logits = self.mlp_head(x)
        
        # Stable sigmoid scaling to [-1, 0] range with gradient clipping
        with torch.no_grad():
            pi_logits.clamp_(-10, 10)  # Prevent extreme values
        
        pi = -torch.sigmoid(pi_logits)  # Maps to [-1, 0]
        pi = torch.clamp(pi, min=-1.0, max=0.0)  # Ensure exact range compliance
        
        return pi

if __name__ == '__main__':
    # Example Usage
    # Assuming a vocabulary and config are available
    # (Need to run vocabulary.py first to create vocab.pkl if using it)
    try:
        from ..utils.vocabulary import Vocabulary
        if os.path.exists(config.VOCAB_PATH):
            vocab = Vocabulary.load_vocab(config.VOCAB_PATH)
            vocab_size = len(vocab)
            print(f"Loaded vocabulary of size: {vocab_size}")
        else:
            print("Vocabulary file not found. Using a dummy vocab_size.")
            vocab_size = 1000 # Dummy vocab size for testing
    except ImportError:
        print("Could not import Vocabulary or config. Using dummy vocab_size.")
        vocab_size = 1000
        # Mock config for direct run if needed
        class MockConfig:
            IMAGE_EMBED_SIZE = 512
            HIDDEN_SIZE = 512
            UE_NUM_LAYERS = 3
            NUM_ATTENTION_HEADS = 8
            FEED_FORWARD_SIZE = 2048
            DROPOUT_PROB = 0.1
        config = MockConfig()

    print(f"Testing UncertaintyEstimator with vocab_size={vocab_size}")

    # Initialize model
    ue_model = UncertaintyEstimator(
        vocab_size=vocab_size,
        image_feature_dim=config.IMAGE_EMBED_SIZE,
        d_model=config.HIDDEN_SIZE,
        num_layers=config.UE_NUM_LAYERS,
        num_heads=config.NUM_ATTENTION_HEADS,
        d_ff=config.FEED_FORWARD_SIZE,
        dropout=config.DROPOUT_PROB
    )
    ue_model.eval() # Set to evaluation mode

    # Create a dummy image feature batch
    batch_size = 4
    dummy_image_features = torch.randn(batch_size, config.IMAGE_EMBED_SIZE)

    # Forward pass
    pi_values = ue_model(dummy_image_features)

    print(f"Input image features shape: {dummy_image_features.shape}")
    print(f"Output pi values shape: {pi_values.shape}")
    assert pi_values.shape == (batch_size, vocab_size)

    # Check if output values are in the range [-1, 0]
    min_val = torch.min(pi_values).item()
    max_val = torch.max(pi_values).item()
    print(f"Min pi value: {min_val:.4f}, Max pi value: {max_val:.4f}")
    assert -1.0001 <= min_val <= 0.0001, f"Min value {min_val} out of range [-1, 0]"
    assert -1.0001 <= max_val <= 0.0001, f"Max value {max_val} out of range [-1, 0]"
    # A small tolerance (0.0001) is added for floating point comparisons.

    print("UncertaintyEstimator tested successfully.") 