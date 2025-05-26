import torch
import torch.nn.functional as F
import heapq

# Robust import for config, vocabulary and models
try:
    from .. import config
    from .vocabulary import Vocabulary
    from ..models.insertion_transformer import InsertionTransformer
    from ..models.uncertainty_estimator import UncertaintyEstimator # Potentially needed for guiding search
except ImportError:
    import sys
    import os
    # Navigate up to the project root (UAIC_NEW) to resolve imports
    current_dir = os.path.dirname(os.path.abspath(__file__)) # utils
    src_dir = os.path.dirname(current_dir) # src
    project_root = os.path.dirname(src_dir) # UAIC_NEW
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if src_dir not in sys.path: # To allow `import config` if in src/
        sys.path.insert(0, src_dir)

    import config # type: ignore
    from utils.vocabulary import Vocabulary # type: ignore
    from models.insertion_transformer import InsertionTransformer # type: ignore
    from models.uncertainty_estimator import UncertaintyEstimator # type: ignore


class BeamHypothesis:
    def __init__(self, sequence: torch.Tensor, score: float, insertions: int):
        self.sequence = sequence
        self.score = float(score)  # 명시적으로 float 타입 강제
        self.insertions = insertions

    def __lt__(self, other):
        return self.score < other.score

def beam_search_inserter(
    image_features: torch.Tensor,
    model: InsertionTransformer,
    vocab: Vocabulary,
    beam_size: int = config.BEAM_SIZE,
    max_iterations: int = config.MAX_SEQ_LEN,
    max_len: int = config.MAX_SEQ_LEN,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> list:
    """
    Performs beam search for an insertion-based captioning model.
    """
    model.eval()
    if image_features.dim() == 1:
        image_features = image_features.unsqueeze(0)

    batch_size = image_features.size(0)
    if batch_size != 1:
        raise ValueError("Beam search for inserter currently supports batch_size=1")

    bos_idx = vocab(config.BOS_TOKEN)
    eos_idx = vocab(config.EOS_TOKEN)
    none_idx = vocab(config.NONE_TOKEN)
    pad_idx = vocab(config.PAD_TOKEN)

    # config에서 정의된 최대 시퀀스 길이 사용
    effective_max_len = min(max_len, config.MAX_SEQ_LEN)

    initial_seq = torch.tensor([bos_idx, eos_idx], dtype=torch.long, device=device)
    beams = [BeamHypothesis(initial_seq, 0.0, 0)]
    completed_hypotheses = []

    for iter_num in range(max_iterations):
        if not beams:
            break

        candidates = []
        for beam in beams:
            # 시퀀스 길이가 제한을 초과하면 완료된 것으로 처리
            current_len = len(beam.sequence) - 2  # BOS와 EOS 토큰 제외
            if current_len >= effective_max_len:
                completed_hypotheses.append(beam)
                continue

            if beam.insertions >= max_iterations:
                completed_hypotheses.append(beam)
                continue

            partial_caption_input = beam.sequence.unsqueeze(0)  # [1, seq_len]
            
            with torch.no_grad():
                word_logits, pos_logits = model(image_features, partial_caption_input)  # [1, seq_len, vocab_size], [1, seq_len]
                
                # 배치 차원 제거
                word_logits = word_logits.squeeze(0)  # [seq_len, vocab_size]
                pos_logits = pos_logits.squeeze(0)    # [seq_len]

                # 확률 계산
                log_word_probs = F.log_softmax(word_logits, dim=-1)  # [seq_len, vocab_size]
                log_pos_probs = F.log_softmax(pos_logits, dim=-1)    # [seq_len]

            # 현재 시퀀스 길이가 최대 길이에 도달하면 새로운 삽입을 허용하지 않음
            if current_len >= effective_max_len - 1:
                continue

            num_possible_slots = len(beam.sequence)
            for slot_idx in range(num_possible_slots):
                slot_score = float(log_pos_probs[slot_idx].item())
                word_scores = log_word_probs[slot_idx]  # [vocab_size]
                
                # top-k 단어 선택
                top_k_scores, top_k_words = torch.topk(word_scores, min(beam_size, len(vocab)))
                
                for word_idx, word_score in zip(top_k_words.tolist(), top_k_scores.tolist()):
                    if word_idx == pad_idx:
                        continue
                        
                    new_score = float(beam.score + slot_score + word_score)
                    s1 = beam.sequence[:slot_idx]
                    s2 = beam.sequence[slot_idx:]
                    word_tensor = torch.tensor([word_idx], device=device, dtype=torch.long)
                    new_seq = torch.cat((s1, word_tensor, s2))
                    
                    # 새로운 시퀀스가 길이 제한을 초과하면 건너뜀
                    if len(new_seq) - 2 > effective_max_len:
                        continue
                        
                    new_insertions = beam.insertions + (1 if word_idx != none_idx else 0)
                    new_beam = BeamHypothesis(new_seq, new_score, new_insertions)
                    
                    if word_idx == none_idx or new_insertions >= max_iterations:
                        completed_hypotheses.append(new_beam)
                    else:
                        candidates.append(new_beam)

        beams = sorted(candidates, reverse=True)[:beam_size]
        completed_hypotheses = sorted(completed_hypotheses, reverse=True)[:beam_size]

    result_sequences = []
    for hyp in sorted(completed_hypotheses, reverse=True):
        seq = hyp.sequence.tolist()
        if seq[0] == bos_idx:
            seq = seq[1:]
        if seq[-1] == eos_idx:
            seq = seq[:-1]
            
        final_tokens = [token for token in seq if token != none_idx]
        
        if final_tokens:
            result_sequences.append(final_tokens)
        elif not result_sequences:
            result_sequences.append([])
            
    if not result_sequences:
        result_sequences.append([])
        
    return result_sequences[:beam_size]


if __name__ == '__main__':
    class MockVocab(Vocabulary):
        def __init__(self):
            super().__init__(special_tokens=[config.PAD_TOKEN, config.BOS_TOKEN, config.EOS_TOKEN, config.UNK_TOKEN, config.NONE_TOKEN, config.MASK_TOKEN])
            for word in ["hello", "world", "a", "cat", "dog"]:
                self.add_word(word)

    class MockInsertionTransformer(nn.Module):
        def __init__(self, vocab_size, d_model, image_feature_dim):
            super().__init__()
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.image_feature_dim = image_feature_dim
            self.word_predictor = nn.Linear(d_model, vocab_size)
            self.pos_predictor = nn.Linear(d_model, 1)
            self.token_embed = nn.Embedding(vocab_size, d_model)
            if image_feature_dim != d_model:
                 self.image_proj = nn.Linear(image_feature_dim, d_model)
            else:
                 self.image_proj = nn.Identity()

        def forward(self, image_features, partial_captions, padding_mask=None):
            batch_size, seq_len = partial_captions.size()
            # Simulate some minimal processing for shape consistency
            # In a real model, this involves embeddings, image feature fusion, and transformer layers.
            # For test purposes, just ensure output shapes are correct.
            if seq_len == 0: # Should ideally not happen with BOS, EOS start
                # If it somehow did, pos_logits for seq_len=0 is problematic.
                # Return empty or handle as error. For now, assume seq_len > 0 due to BOS/EOS.
                 dummy_transformer_output = torch.randn(batch_size, 1, self.d_model, device=partial_captions.device)
                 # word_logits for 1 position, pos_logits for 1 position
                 word_logits = self.word_predictor(dummy_transformer_output)
                 pos_logits = self.pos_predictor(dummy_transformer_output).squeeze(-1)
            else:
                dummy_transformer_output = torch.randn(batch_size, seq_len, self.d_model, device=partial_captions.device)
                word_logits = self.word_predictor(dummy_transformer_output) 
                pos_logits = self.pos_predictor(dummy_transformer_output).squeeze(-1)
            return word_logits, pos_logits

    print("Testing beam_search_inserter...")
    device = torch.device("cpu")

    mock_vocab = MockVocab()
    vocab_size = len(mock_vocab)
    # Use smaller dimensions for faster testing if config values are large
    d_model_test = getattr(config, 'HIDDEN_SIZE', 64)
    img_feat_dim_test = getattr(config, 'IMAGE_EMBED_SIZE', 64)
    
    mock_model = MockInsertionTransformer(vocab_size, d_model_test, img_feat_dim_test).to(device)
    mock_model.eval()

    dummy_image_features = torch.randn(1, img_feat_dim_test, device=device)

    test_beam_size = 3
    test_max_iterations = 5 
    test_max_len = 7

    print(f"Vocab size: {vocab_size}")
    print(f"BOS: {mock_vocab(config.BOS_TOKEN)}, EOS: {mock_vocab(config.EOS_TOKEN)}, NONE: {mock_vocab(config.NONE_TOKEN)}")

    generated_captions_tokens = beam_search_inserter(
        image_features=dummy_image_features,
        model=mock_model,
        vocab=mock_vocab,
        beam_size=test_beam_size,
        max_iterations=test_max_iterations,
        max_len=test_max_len,
        device=device
    )

    print(f"Generated {len(generated_captions_tokens)} captions:")
    for i, token_ids in enumerate(generated_captions_tokens):
        caption_text = " ".join([mock_vocab.get_word(token_id) for token_id in token_ids])
        # Escape triple quotes for the print f-string
        print(f'  {i+1}: {token_ids} -> "{caption_text.replace(""", "\\""")}"')

    assert len(generated_captions_tokens) <= test_beam_size
    if generated_captions_tokens:
        for tokens in generated_captions_tokens:
            # An empty list is a valid result (e.g., if only BOS, NONE, EOS was generated)
            if tokens: # Only check length and content if tokens list is not empty
                assert len(tokens) <= test_max_len
                assert mock_vocab(config.BOS_TOKEN) not in tokens
                assert mock_vocab(config.EOS_TOKEN) not in tokens
                assert mock_vocab(config.NONE_TOKEN) not in tokens
                for token_id in tokens:
                    assert 0 <= token_id < vocab_size
            else: # tokens is an empty list
                assert len(tokens) == 0


    # Test case: Model always predicts NONE token first
    class MockInsertionTransformerPredictsNone(MockInsertionTransformer):
        def forward(self, image_features, partial_captions, padding_mask=None):
            batch_size, seq_len = partial_captions.size()
            none_idx = mock_vocab(config.NONE_TOKEN)
            
            word_logits_val = torch.full((batch_size, seq_len, vocab_size), -10.0, device=device)
            if seq_len > 0:
                # Make NONE highly probable for the first slot only to simplify test
                word_logits_val[:, 0, none_idx] = 10.0 
            
            # Make the first position slot highly probable
            pos_logits_val = torch.full((batch_size, seq_len), -10.0, device=device)
            if seq_len > 0:
                pos_logits_val[:, 0] = 10.0

            return word_logits_val, pos_logits_val
            
    mock_model_none = MockInsertionTransformerPredictsNone(vocab_size, d_model_test, img_feat_dim_test).to(device)
    mock_model_none.eval()

    print("\nTesting with model that predicts [NONE] immediately...")
    generated_captions_none = beam_search_inserter(
        image_features=dummy_image_features,
        model=mock_model_none,
        vocab=mock_vocab,
        beam_size=test_beam_size,
        max_iterations=test_max_iterations,
        max_len=test_max_len,
        device=device
    )
    print(f"Generated {len(generated_captions_none)} captions (expecting one empty list):")
    for i, token_ids in enumerate(generated_captions_none):
        caption_text = " ".join([mock_vocab.get_word(token_id) for token_id in token_ids])
        print(f'  {i+1}: {token_ids} -> "{caption_text.replace(""", "\\""")}"') # Escape triple quotes
    
    # Expect one result, which is an empty list [].
    assert len(generated_captions_none) == 1, f"Expected 1 caption, got {len(generated_captions_none)}"
    assert generated_captions_none[0] == [], f"Expected an empty list, got {generated_captions_none[0]}"

    print("\nBeam search inserter tests completed successfully.") 