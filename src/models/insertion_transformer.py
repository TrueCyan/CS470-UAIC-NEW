import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

# Robust import for config and transformer_blocks
try:
    from .. import config
    # Ensure Vocabulary can be imported for MockVocab in __main__
    from ..utils.vocabulary import Vocabulary
    from .transformer_blocks import TransformerEncoderLayer, PositionalEncoding
except ImportError:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir) # src/
    project_root = os.path.dirname(parent_dir) # UAIC_NEW/
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if parent_dir not in sys.path: # To allow `import config` if in src/
        sys.path.insert(0, parent_dir)
    
    import config # type: ignore
    from models.transformer_blocks import TransformerEncoderLayer, PositionalEncoding # type: ignore
    # Need Vocabulary for MockVocab if running directly from models directory
    try:
        from utils.vocabulary import Vocabulary # type: ignore
    except ImportError: 
        sys.path.append(os.path.join(project_root, 'src')) # Ensure src is in path
        from utils.vocabulary import Vocabulary # type: ignore


class InsertionTransformer(nn.Module):
    def __init__(self, vocab_size: int, 
                 d_model: int = config.HIDDEN_SIZE, 
                 num_layers: int = config.IT_NUM_LAYERS, 
                 num_heads: int = config.NUM_ATTENTION_HEADS, 
                 d_ff: int = config.FEED_FORWARD_SIZE, 
                 dropout: float = config.DROPOUT_PROB, 
                 image_feature_dim: int = config.IMAGE_EMBED_SIZE,
                 max_seq_len: int = config.MAX_SEQ_LEN,
                 pad_idx: int = 0): 
        super(InsertionTransformer, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_seq_len)

        if image_feature_dim != d_model:
            self.image_proj = nn.Linear(image_feature_dim, d_model)
        else:
            self.image_proj = nn.Identity()

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.word_predictor = nn.Linear(d_model, vocab_size)
        self.pos_predictor = nn.Linear(d_model, 1) 
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, image_features: torch.Tensor, 
                partial_captions: torch.Tensor, 
                padding_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = partial_captions.size()

        cap_embed = self.token_embed(partial_captions) 
        cap_embed = self.pos_encoder(cap_embed * math.sqrt(self.d_model)) 
        cap_embed = self.dropout_layer(cap_embed)

        img_feat_proj = self.image_proj(image_features).unsqueeze(1)
        combined_input = cap_embed + img_feat_proj 

        if padding_mask is None:
            padding_mask = (partial_captions != self.pad_idx)
        
        transformer_mask = padding_mask

        transformer_output = combined_input
        for layer_module in self.transformer_layers:
            transformer_output = layer_module(transformer_output, src_mask=transformer_mask)

        word_logits = self.word_predictor(transformer_output)
        pos_logits = self.pos_predictor(transformer_output).squeeze(-1)

        return word_logits, pos_logits

if __name__ == '__main__':
    class MockVocabForIT(Vocabulary):
        def __init__(self):
            super().__init__()
            self.built_custom_tokens = False

        def build_mock_vocab(self, mock_cfg_tokens):
            test_tokens = ["a", "cat", "sat", "on", "the", "mat",
                           mock_cfg_tokens.MASK_TOKEN, mock_cfg_tokens.PAD_TOKEN, 
                           mock_cfg_tokens.BOS_TOKEN, mock_cfg_tokens.EOS_TOKEN, 
                           mock_cfg_tokens.NONE_TOKEN, mock_cfg_tokens.UNK_TOKEN]
            for word in test_tokens:
                if word not in self.word2idx:
                    self.add_word(word)
            self.built_custom_tokens = True

        def _ensure_built(self):
            if not self.built_custom_tokens:
                class MinimalTokens:
                    PAD_TOKEN="[PAD]"; MASK_TOKEN="[MASK]"; BOS_TOKEN="[BOS]"; EOS_TOKEN="[EOS]"; NONE_TOKEN="[NONE]"; UNK_TOKEN="[UNK]"
                self.build_mock_vocab(MinimalTokens())

        def __call__(self, word):
            self._ensure_built()
            return super().__call__(word)

        def get_word(self, idx):
            self._ensure_built()
            return super().get_word(idx)

        def __len__(self):
            self._ensure_built()
            return super().__len__()

    class MockConfigForITTest:
        HIDDEN_SIZE = 64
        IT_NUM_LAYERS = 2
        NUM_ATTENTION_HEADS = 2
        FEED_FORWARD_SIZE = 128
        DROPOUT_PROB = 0.1
        IMAGE_EMBED_SIZE = 64
        MAX_SEQ_LEN = 20
        PAD_TOKEN = "[PAD]"
        BOS_TOKEN = "[BOS]"
        EOS_TOKEN = "[EOS]"
        MASK_TOKEN = "[MASK]"
        NONE_TOKEN = "[NONE]"
        UNK_TOKEN = "[UNK]"
        test_vocab_instance = MockVocabForIT()
        
        def __init__(self):
            self.test_vocab_instance.build_mock_vocab(self)

        def get_pad_idx(self):
            return self.test_vocab_instance(self.PAD_TOKEN)
        
        def get_vocab_size(self):
            return len(self.test_vocab_instance)

    test_config_instance = MockConfigForITTest()
    VOCAB_SIZE_FOR_TEST = test_config_instance.get_vocab_size()
    PAD_ID_FOR_TEST = test_config_instance.get_pad_idx()

    print(f"Testing InsertionTransformer with vocab_size={VOCAB_SIZE_FOR_TEST}")
    print(f"PAD ID for embedding: {PAD_ID_FOR_TEST}")

    it_model = InsertionTransformer(
        vocab_size=VOCAB_SIZE_FOR_TEST,
        d_model=test_config_instance.HIDDEN_SIZE,
        num_layers=test_config_instance.IT_NUM_LAYERS,
        num_heads=test_config_instance.NUM_ATTENTION_HEADS,
        d_ff=test_config_instance.FEED_FORWARD_SIZE,
        dropout=test_config_instance.DROPOUT_PROB,
        image_feature_dim=test_config_instance.IMAGE_EMBED_SIZE,
        max_seq_len=test_config_instance.MAX_SEQ_LEN,
        pad_idx=PAD_ID_FOR_TEST
    )
    it_model.eval()

    batch_size = 2
    seq_len_for_test = test_config_instance.MAX_SEQ_LEN - 5 
    dummy_image_features = torch.randn(batch_size, test_config_instance.IMAGE_EMBED_SIZE)
    
    dummy_partial_captions = torch.full((batch_size, seq_len_for_test), PAD_ID_FOR_TEST, dtype=torch.long)
    for r in range(batch_size):
        for c in range(seq_len_for_test - 2): 
            random_word_id = torch.randint(0, VOCAB_SIZE_FOR_TEST, (1,)).item()
            dummy_partial_captions[r, c] = random_word_id
    
    explicit_padding_mask = (dummy_partial_captions != PAD_ID_FOR_TEST)

    word_logits, pos_logits = it_model(dummy_image_features, dummy_partial_captions, padding_mask=explicit_padding_mask)

    print(f"Input image_features shape: {dummy_image_features.shape}")
    print(f"Input partial_captions shape: {dummy_partial_captions.shape}")
    print(f"Output word_logits shape: {word_logits.shape}")
    print(f"Output pos_logits shape: {pos_logits.shape}")

    assert word_logits.shape == (batch_size, seq_len_for_test, VOCAB_SIZE_FOR_TEST)
    assert pos_logits.shape == (batch_size, seq_len_for_test)

    word_logits_no_mask, pos_logits_no_mask = it_model(dummy_image_features, dummy_partial_captions, padding_mask=None)
    assert word_logits_no_mask.shape == (batch_size, seq_len_for_test, VOCAB_SIZE_FOR_TEST)
    assert pos_logits_no_mask.shape == (batch_size, seq_len_for_test)

    print("InsertionTransformer tested successfully.") 