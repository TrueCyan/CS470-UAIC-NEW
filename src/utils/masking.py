# DP 기반 마스킹 알고리즘 구현 예정 

import torch
import random
import numpy as np
from typing import List, Tuple, Dict

# Robust import for config and Vocabulary
try:
    from .. import config
    from .vocabulary import Vocabulary # For token indices
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
    from utils.vocabulary import Vocabulary

def generate_insertion_training_samples(
    true_caption_indices: List[int], 
    uncertainty_scores: torch.Tensor, # Shape: (vocab_size) or (seq_len_of_caption)
    vocab: Vocabulary,
    eos_id: int,
    bos_id: int,
    pad_id: int,
    mask_id: int,
    none_id: int,
    max_len: int = config.MAX_SEQ_LEN
) -> List[Tuple[List[int], int, int]]:
    """
    Generates a sequence of training samples (S_k-1, target_word, target_pos)
    for the insertion transformer based on uncertainty scores.
    Words with lower uncertainty (pi closer to 0, or -pi smaller) are revealed earlier.

    Args:
        true_caption_indices: Ground truth caption, list of token IDs (excluding BOS/EOS initially).
        uncertainty_scores: Tensor of uncertainty scores (pi, range [-1, 0]) for each word in vocab.
                              Or, if pre-computed for words in *this specific caption*, a tensor of shape (len(true_caption_indices)).
        vocab: Vocabulary object.
        eos_id, bos_id, pad_id, mask_id, none_id: Special token IDs.
        max_len: Maximum sequence length for S_k-1.

    Returns:
        A list of tuples: (S_k_minus_1_padded, target_word_id, target_pos_idx).
        S_k_minus_1_padded: The source sequence for the transformer with MASK tokens, BOS/EOS, and padding.
        target_word_id: The ID of the word to be inserted.
        target_pos_idx: The index in S_k_minus_1 (0 to len(S_k_minus_1)-1) *before which* the target word should be inserted.
                        (i.e., if target_pos_idx is i, word is inserted between S_k_minus_1[i-1] and S_k_minus_1[i]).
                        The final slot (after the last visible word but before EOS) is also a valid insertion position.
    """
    training_samples = []
    
    # Filter out BOS/EOS/PAD from true_caption_indices if they are present
    # The core logic works on the actual content words.
    content_caption = [idx for idx in true_caption_indices if idx not in [bos_id, eos_id, pad_id]]
    if not content_caption:
        return []
    
    num_content_words = len(content_caption)

    # Get uncertainty for words in the current caption.
    # We want to reveal words with pi closer to 0 first (less uncertain).
    # So we sort by pi in ascending order (e.g., -0.1 is revealed before -0.9).
    # Or, if psi = -pi, sort by psi in descending order.
    # Let's assume uncertainty_scores are pi values for this caption's words.
    
    word_uncertainties = []
    if uncertainty_scores.dim() == 1 and uncertainty_scores.size(0) == num_content_words:
        # Scores are pre-calculated for words in this caption
        for i in range(num_content_words):
            word_uncertainties.append((content_caption[i], uncertainty_scores[i].item()))
    elif uncertainty_scores.dim() == 1 and uncertainty_scores.size(0) == len(vocab):
        # Scores are for the whole vocabulary
        for i in range(num_content_words):
            word_id = content_caption[i]
            # Ensure word_id is a valid index for uncertainty_scores
            if 0 <= word_id < len(vocab):
                 word_uncertainties.append((word_id, uncertainty_scores[word_id].item()))
            else: # Should not happen with a correctly built vocab and dataset
                 word_uncertainties.append((word_id, -0.5)) # Assign average uncertainty for OOV in this context
    else:
        raise ValueError("Invalid uncertainty_scores shape")

    # Sort words by their uncertainty (pi: -1 (high) to 0 (low)). We want to reveal low uncertainty first.
    # So, sort by pi in descending order (0 comes before -1).
    # sorted_words_to_reveal = sorted(word_uncertainties, key=lambda x: x[1], reverse=True)
    # Let's re-verify: low uncertainty = pi close to 0. High uncertainty = pi close to -1.
    # We want to reveal words with low uncertainty first.
    # So, if pi = -0.1 (low unc) and pi = -0.9 (high unc), -0.1 should come before -0.9.
    # This means sorting in descending order of pi values (e.g. -0.1 > -0.9).
    sorted_word_indices_to_reveal = sorted(range(num_content_words), 
                                           key=lambda i: word_uncertainties[i][1], 
                                           reverse=True)

    # S_k represents the sequence with k content words revealed.
    # It always starts with BOS and aims to end with EOS.
    # current_sk always includes BOS and EOS, with MASKs in between.
    
    # Initial state S_0: [BOS, MASK, ..., MASK, EOS] (num_content_words MASKs)
    # Or, following the paper, S_0 = [BOS, EOS]
    # Let's start with S_0 = [BOS, EOS] as a base for insertions. Max k steps = num_content_words.
    # At step k (0 to num_content_words-1), we generate S_k -> S_k+1
    # S_k is the partially revealed sequence (source for transformer)
    # The target is the (k+1)-th word and its insertion position in S_k.

    current_revealed_sequence = [bos_id, eos_id] # S_0

    # Store the original indices of revealed words to maintain order for insertion
    # This list will store (original_index_in_content_caption, word_id)
    revealed_content_with_orig_indices = []

    for k in range(num_content_words):
        word_orig_idx_to_reveal = sorted_word_indices_to_reveal[k]
        word_id_to_reveal = content_caption[word_orig_idx_to_reveal]

        # S_k_minus_1 is `current_revealed_sequence` from previous step (or S_0 initially)
        # This is the input to the insertion transformer.
        # Pad S_k_minus_1 to max_len for the transformer input
        s_k_minus_1_padded = current_revealed_sequence[:]
        s_k_minus_1_padded.extend([pad_id] * (max_len - len(s_k_minus_1_padded)))
        s_k_minus_1_padded = s_k_minus_1_padded[:max_len] # Ensure it does not exceed max_len
        
        # Determine target_pos_idx: where to insert word_id_to_reveal into current_revealed_sequence
        # to maintain the original relative order of revealed words.
        # `revealed_content_with_orig_indices` is sorted by original index.
        # Find the correct slot in `current_revealed_sequence` (which is also sorted by orig_idx).
        # `current_revealed_sequence` = [BOS, w_a, w_b, EOS] if w_a (orig_idx i) and w_b (orig_idx j) revealed (i < j).
        
        insert_before_this_idx_in_current = len(current_revealed_sequence) - 1 # Default: insert before EOS
        for i, (orig_idx, _) in enumerate(revealed_content_with_orig_indices):
            if word_orig_idx_to_reveal < orig_idx:
                insert_before_this_idx_in_current = i + 1 # +1 because current_revealed_sequence has BOS at index 0
                break
        
        target_pos_idx = insert_before_this_idx_in_current # This is the insertion slot index in current_revealed_sequence
        target_word_id = word_id_to_reveal
        
        # Add the sample: (S_k-1, word_to_insert, position_to_insert_before)
        training_samples.append((s_k_minus_1_padded, target_word_id, target_pos_idx))

        # Construct S_k (the next state) by inserting the newly revealed word
        current_revealed_sequence.insert(target_pos_idx, word_id_to_reveal)
        revealed_content_with_orig_indices.append((word_orig_idx_to_reveal, word_id_to_reveal))
        revealed_content_with_orig_indices.sort(key=lambda x: x[0]) # Keep sorted by original index

        # Constraint: The total length of current_revealed_sequence should not exceed max_len
        # This is implicitly handled if num_content_words + 2 <= max_len.
        # If not, the padded sequence might be truncated, which is fine. Insertion Transformer should handle it.

    # Add a final sample for predicting [NONE] if the sequence is complete or max_len reached.
    # This helps the model learn to stop inserting.
    # S_N (all words revealed) as input, target is [NONE] at a chosen position (e.g., after last real word).
    # If current_revealed_sequence already hit max_len, this step might be tricky.
    if len(current_revealed_sequence) < max_len: # Only if there's space to predict NONE
        s_N_padded = current_revealed_sequence[:]
        s_N_padded.extend([pad_id] * (max_len - len(s_N_padded)))
        s_N_padded = s_N_padded[:max_len]
        
        # Predict [NONE] at the end (before EOS)
        # The number of valid insertion slots in s_N_padded is len(current_revealed_sequence). BOS s1 s2 EOS -> slots: _ BOS _ s1 _ s2 _ EOS
        # Let's choose the slot just before EOS as the target position for [NONE]
        # Position is len(current_revealed_sequence) - 1 in the original `current_revealed_sequence` before padding.
        # This is the slot before EOS. For [BOS, w1, w2, EOS], this is index 3.
        target_pos_for_none = len(current_revealed_sequence) -1 
        if target_pos_for_none < 0 : target_pos_for_none = 0 # safety for empty seq
        training_samples.append((s_N_padded, none_id, target_pos_for_none))

    return training_samples

if __name__ == '__main__':
    # --- Setup for testing --- 
    class MockVocab(Vocabulary): # Extend the actual Vocabulary for testing
        def __init__(self):
            super().__init__()
            self.built = False
        def build_mock_vocab(self, words=None):
            if words is None:
                words = ["a", "cat", "sat", "on", "the", "mat", "dog", "ran"]
            # Add special tokens first (already done in Vocabulary.__init__)
            # self.add_word(config.PAD_TOKEN)
            # self.add_word(config.BOS_TOKEN)
            # self.add_word(config.EOS_TOKEN)
            # self.add_word(config.UNK_TOKEN)
            # self.add_word(config.NONE_TOKEN)
            # self.add_word(config.MASK_TOKEN) # MASK_TOKEN added to config
            self.add_word(config.MASK_TOKEN) # Ensure MASK is in vocab for tests

            for word in words:
                self.add_word(word)
            self.built = True
        def __call__(self, word):
            if not self.built: self.build_mock_vocab()
            return super().__call__(word)
        def get_word(self, idx):
            if not self.built: self.build_mock_vocab()
            return super().get_word(idx)
        def __len__(self):
            if not self.built: self.build_mock_vocab()
            return super().__len__()

    vocab = MockVocab()
    vocab.build_mock_vocab()

    bos_id = vocab(config.BOS_TOKEN)
    eos_id = vocab(config.EOS_TOKEN)
    pad_id = vocab(config.PAD_TOKEN)
    mask_id = vocab(config.MASK_TOKEN)
    none_id = vocab(config.NONE_TOKEN)
    unk_id = vocab(config.UNK_TOKEN)

    print(f"Vocab size: {len(vocab)}")
    print(f"BOS: {bos_id}, EOS: {eos_id}, PAD: {pad_id}, MASK: {mask_id}, NONE: {none_id}, UNK: {unk_id}")

    # Example 1: Simple caption
    caption_text = "a cat sat on the mat"
    true_caption_indices = [vocab(token) for token in caption_text.split()]
    # Dummy uncertainty scores (pi values in [-1, 0]) for words in this caption
    # Lower uncertainty (closer to 0) should be revealed first.
    # "a" (low), "cat" (low), "sat" (mid), "on" (high), "the" (mid), "mat" (low)
    # So order might be: a, cat, mat, sat, the, on
    # Let pi be: a:-0.1, cat:-0.15, sat:-0.5, on:-0.8, the:-0.4, mat:-0.2
    # Sorted by pi descending: a, cat, mat, the, sat, on
    caption_uncertainties = torch.tensor([-0.1, -0.15, -0.5, -0.8, -0.4, -0.2])
    
    print(f"\n--- Example 1: '{caption_text}' ---")
    print(f"True indices: {true_caption_indices}")
    print(f"Uncertainties (pi): {caption_uncertainties.tolist()}")

    samples = generate_insertion_training_samples(
        true_caption_indices,
        caption_uncertainties,
        vocab,
        eos_id, bos_id, pad_id, mask_id, none_id,
        max_len = 10 # Short max_len for display
    )

    print(f"Generated {len(samples)} training samples:")
    for i, (s_k_minus_1, target_word, target_pos) in enumerate(samples):
        s_k_minus_1_text = " ".join([vocab.get_word(idx) for idx in s_k_minus_1 if idx != pad_id])
        target_word_text = vocab.get_word(target_word)
        print(f"  Sample {i+1}: ")
        print(f"    S_k-1 (len {len(s_k_minus_1)}): {s_k_minus_1_text}  {s_k_minus_1}")
        print(f"    Target word: {target_word_text} ({target_word}) at pos_idx: {target_pos}")

    # Example 2: Uncertainty scores from full vocabulary
    caption_text_2 = "dog ran fast"
    true_caption_indices_2 = [vocab(token) for token in caption_text_2.split()]
    # Assume vocab indices for dog, ran, fast are 10, 11, 12 respectively
    # Let their pi be -0.2, -0.7, -0.4
    # Expected reveal order: dog (-0.2), fast (-0.4), ran (-0.7)
    full_vocab_uncertainties = torch.full((len(vocab),), -0.5) # Default uncertainty
    if vocab("dog") < len(vocab): full_vocab_uncertainties[vocab("dog")] = -0.2
    if vocab("ran") < len(vocab): full_vocab_uncertainties[vocab("ran")] = -0.7
    if vocab("fast") < len(vocab): full_vocab_uncertainties[vocab("fast")] = -0.4
    
    print(f"\n--- Example 2: '{caption_text_2}' (uncertainty from full vocab) ---")
    samples_2 = generate_insertion_training_samples(
        true_caption_indices_2,
        full_vocab_uncertainties,
        vocab,
        eos_id, bos_id, pad_id, mask_id, none_id,
        max_len=10
    )
    print(f"Generated {len(samples_2)} training samples:")
    for i, (s_k_minus_1, target_word, target_pos) in enumerate(samples_2):
        s_k_minus_1_text = " ".join([vocab.get_word(idx) for idx in s_k_minus_1 if idx != pad_id])
        target_word_text = vocab.get_word(target_word)
        print(f"  Sample {i+1}: S_k-1: '{s_k_minus_1_text}', Target: '{target_word_text}' at pos {target_pos}")

    print("\nMasking utility tested.") 