# 손실 함수 구현 예정 

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

# Robust import for config
try:
    from .. import config
except ImportError:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(src_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import config # type: ignore

def compute_insertion_losses(
    word_logits: torch.Tensor,      # (batch_size, seq_len, vocab_size)
    pos_logits: torch.Tensor,       # (batch_size, seq_len)
    target_word_ids: torch.Tensor,  # (batch_size)
    target_pos_indices: torch.Tensor, # (batch_size)
    pad_idx: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the NLL_word and NLL_pos losses for the insertion transformer.

    Args:
        word_logits: Logits for word prediction from the model.
        pos_logits: Logits for position prediction from the model.
        target_word_ids: Ground truth word IDs to be inserted.
        target_pos_indices: Ground truth positions where words should be inserted.
        pad_idx: Padding index, to be ignored in loss calculation (though typically
                 target_pos_indices should point to non-padded locations).

    Returns:
        word_loss: Negative log-likelihood loss for word prediction.
        pos_loss: Negative log-likelihood loss for position prediction.
        total_loss: Sum of word_loss and pos_loss.
    """
    batch_size = word_logits.size(0)

    # 1. Word Prediction Loss (NLL_word)
    # We need to select the word logits at the target_pos_indices for each batch item.
    # word_logits: (B, S, V)
    # target_pos_indices: (B) values from 0 to S-1
    # gathered_word_logits should be (B, V)
    
    # Create an index tensor for gather compatible with target_pos_indices
    # It needs to be (B, 1, V) to select one S dim vector, and then squeeze, or (B, V) directly if possible.
    # Easier: use advanced indexing or a loop (less efficient but clear for small batches or understanding)
    # Efficient way: expand target_pos_indices to match word_logits dimensions for gather
    idx_exp = target_pos_indices.view(batch_size, 1, 1).expand(-1, -1, word_logits.size(2))
    # idx_exp is (B, 1, V). It selects the S-dimension according to target_pos_indices
    # gathered_word_logits will be (B, 1, V)
    gathered_word_logits = word_logits.gather(1, idx_exp).squeeze(1) # (B, V)

    # 특수 토큰에 대한 마스킹 및 가중치 적용
    special_token_weight = getattr(config, 'SPECIAL_TOKEN_WEIGHT', 0.1)  # 기본값 0.1
    word_loss_weight = getattr(config, 'WORD_LOSS_WEIGHT', 1.0)  # 기본값 1.0
    pos_loss_weight = getattr(config, 'POS_LOSS_WEIGHT', 1.0)  # 기본값 1.0
    
    word_loss = F.cross_entropy(gathered_word_logits, target_word_ids)
    
    # 2. Position Prediction Loss (NLL_pos)
    # pos_logits: (B, S) - scores for each of the S slots
    # target_pos_indices: (B) - the correct slot index (0 to S-1)
    # 시퀀스 길이에 따른 정규화를 위한 가중치
    seq_length = pos_logits.size(1)
    pos_weight = 1.0 / math.sqrt(seq_length)
    
    pos_loss = F.cross_entropy(pos_logits, target_pos_indices) * pos_weight

    # 3. Total Loss with weighted combination
    total_loss = word_loss_weight * word_loss + pos_loss_weight * pos_loss

    return word_loss, pos_loss, total_loss

if __name__ == '__main__':
    print("Testing compute_insertion_losses...")

    # Mock a vocabulary and config for testing
    class MockVocab:
        def __init__(self):
            self.pad_idx = 0
            self.bos_idx = 1
            self.eos_idx = 2
            self.none_idx = 3
            self.word_start_idx = 4
            self.vocab_size = 10 # PAD, BOS, EOS, NONE, 6 other words
        def __call__(self, token_str):
            if token_str == config.PAD_TOKEN: return self.pad_idx
            if token_str == config.BOS_TOKEN: return self.bos_idx
            if token_str == config.EOS_TOKEN: return self.eos_idx
            if token_str == config.NONE_TOKEN: return self.none_idx
            return self.word_start_idx # dummy
        def size(self):
            return self.vocab_size

    mock_vocab = MockVocab()
    # Ensure config tokens are usable if directly referenced by MockVocab or losses
    if not hasattr(config, 'PAD_TOKEN'): config.PAD_TOKEN = "[PAD]"
    if not hasattr(config, 'BOS_TOKEN'): config.BOS_TOKEN = "[BOS]"
    if not hasattr(config, 'EOS_TOKEN'): config.EOS_TOKEN = "[EOS]"
    if not hasattr(config, 'NONE_TOKEN'): config.NONE_TOKEN = "[NONE]"


    batch_size_test = 4
    seq_len_test = 5 # Number of possible insertion slots (e.g., length of S_k-1)
    vocab_size_test = mock_vocab.size()
    pad_idx_test = mock_vocab.pad_idx

    # Dummy model outputs
    # word_logits: (B, S, V)
    dummy_word_logits = torch.randn(batch_size_test, seq_len_test, vocab_size_test)
    # pos_logits: (B, S)
    dummy_pos_logits = torch.randn(batch_size_test, seq_len_test)

    # Dummy ground truth targets
    # target_word_ids: (B) - IDs of words to be inserted (e.g., not PAD)
    # Values should be < vocab_size_test and not pad_idx_test (unless explicitly testing pad ignoring)
    dummy_target_word_ids = torch.randint(mock_vocab.bos_idx, vocab_size_test, (batch_size_test,), dtype=torch.long)
    
    # target_pos_indices: (B) - indices of slots where words are inserted
    # Values should be < seq_len_test
    dummy_target_pos_indices = torch.randint(0, seq_len_test, (batch_size_test,), dtype=torch.long)

    print(f"Word Logits shape: {dummy_word_logits.shape}")
    print(f"Pos Logits shape: {dummy_pos_logits.shape}")
    print(f"Target Word IDs shape: {dummy_target_word_ids.shape}")
    print(f"Target Pos Indices shape: {dummy_target_pos_indices.shape}")

    # Compute losses
    word_loss_val, pos_loss_val, total_loss_val = compute_insertion_losses(
        dummy_word_logits,
        dummy_pos_logits,
        dummy_target_word_ids,
        dummy_target_pos_indices,
        pad_idx_test
    )

    print(f"Word Loss: {word_loss_val.item()}")
    print(f"Position Loss: {pos_loss_val.item()}")
    print(f"Total Loss: {total_loss_val.item()}")

    assert word_loss_val.item() >= 0
    assert pos_loss_val.item() >= 0
    assert total_loss_val.item() >= 0
    assert torch.isclose(total_loss_val, config.WORD_LOSS_WEIGHT * word_loss_val + config.POS_LOSS_WEIGHT * pos_loss_val)

    # Test with target_word_id = NONE_TOKEN
    target_word_is_none = torch.full_like(dummy_target_word_ids, mock_vocab.none_idx)
    wl_none, pl_none, tl_none = compute_insertion_losses(
        dummy_word_logits, dummy_pos_logits, target_word_is_none, dummy_target_pos_indices, pad_idx_test
    )
    print(f"Word Loss (target NONE): {wl_none.item()}")
    assert wl_none.item() >=0

    # Test case: Check if loss decreases when logits for target are high
    high_logit_word_logits = torch.full((batch_size_test, seq_len_test, vocab_size_test), -10.0)
    high_logit_pos_logits = torch.full((batch_size_test, seq_len_test), -10.0)

    for i in range(batch_size_test):
        target_pos = dummy_target_pos_indices[i].item()
        target_word = dummy_target_word_ids[i].item()
        high_logit_word_logits[i, target_pos, target_word] = 10.0 # High logit for correct word at correct pos
        high_logit_pos_logits[i, target_pos] = 10.0 # High logit for correct pos

    wl_low, pl_low, tl_low = compute_insertion_losses(
        high_logit_word_logits, high_logit_pos_logits, 
        dummy_target_word_ids, dummy_target_pos_indices, pad_idx_test
    )
    print(f"Word Loss (low, high target logit): {wl_low.item()}")
    print(f"Position Loss (low, high target logit): {pl_low.item()}")
    print(f"Total Loss (low, high target logit): {tl_low.item()}")
    assert wl_low.item() < word_loss_val.item() # Expect lower loss
    assert pl_low.item() < pos_loss_val.item() # Expect lower loss

    print("compute_insertion_losses tests passed.") 