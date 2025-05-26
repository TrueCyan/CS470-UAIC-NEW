# 트랜스포머 기본 블록 구현 예정 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple # Added for type hinting

# Robust import for config
try:
    from .. import config
except ImportError:
    import sys
    import os
    # Add the project root to sys.path if running this script directly for testing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir) # This should be src/
    project_root = os.path.dirname(parent_dir) # This should be UAIC_NEW/
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if parent_dir not in sys.path:
         sys.path.insert(0, parent_dir) 
    import config

class PositionalEncoding(nn.Module):
    """Implement the PE function."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = config.MAX_SEQ_LEN):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension: [1, max_len, d_model]
        self.register_buffer('pe', pe) # So it moves to device with the model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # 시퀀스 길이가 위치 인코딩의 최대 길이를 초과하지 않도록 제한
        seq_len = min(x.size(1), self.pe.size(1))
        x = x[:, :seq_len, :]
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q (torch.Tensor): Query tensor, shape (batch_size, num_heads, seq_len_q, d_k)
            K (torch.Tensor): Key tensor, shape (batch_size, num_heads, seq_len_k, d_k)
            V (torch.Tensor): Value tensor, shape (batch_size, num_heads, seq_len_v, d_k) (seq_len_k == seq_len_v)
            mask (torch.Tensor, optional): Mask tensor, shape (batch_size, 1, 1, seq_len_k) or (batch_size, 1, seq_len_q, seq_len_k).
                                          Defaults to None.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """
        batch_size = Q.size(0)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = scores / math.sqrt(self.d_k)  # Scale by sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            # 마스크의 크기를 scores와 맞추기
            if mask.size(-1) != scores.size(-1):
                # 마스크가 더 작은 경우, 패딩
                if mask.size(-1) < scores.size(-1):
                    pad_size = scores.size(-1) - mask.size(-1)
                    mask = F.pad(mask, (0, pad_size), value=True)
                # 마스크가 더 큰 경우, 자르기
                else:
                    mask = mask[..., :scores.size(-1)]
            
            # 마스크 적용
            scores = scores.masked_fill(mask == 0, -65504.0)  # FP16에서 처리 가능한 최소값 사용
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len_q, d_v)
        
        return output, attn_weights

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query (torch.Tensor): (batch_size, seq_len_q, d_model)
            key (torch.Tensor): (batch_size, seq_len_k, d_model)
            value (torch.Tensor): (batch_size, seq_len_v, d_model) (seq_len_k == seq_len_v)
            mask (torch.Tensor, optional): Mask to be applied. Defaults to None.
                                         Shape could be (batch_size, seq_len_q, seq_len_k) or (batch_size, 1, seq_len_k).
                                         It will be reshaped for multi-head attention.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The attention output tensor (batch_size, seq_len_q, d_model) and attention weights.
        """
        batch_size = query.size(0)

        # 1) Linear projections
        Q = self.q_linear(query)  # (batch_size, seq_len_q, d_model)
        K = self.k_linear(key)    # (batch_size, seq_len_k, d_model)
        V = self.v_linear(value)  # (batch_size, seq_len_v, d_model)

        # 2) Reshape for multi-head: (batch_size, seq_len, num_heads, d_k) and transpose to (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3) Apply scaled dot-product attention
        # If mask is (batch_size, seq_len_q, seq_len_k) or (batch_size, 1, seq_len_k) for padding,
        # it needs to be (batch_size, 1, seq_len_q, seq_len_k) or (batch_size, 1, 1, seq_len_k) for broadcasting.
        if mask is not None:
            # Ensure mask is 4D for broadcasting with (batch_size, num_heads, seq_len_q, seq_len_k)
            if mask.dim() == 2: # (batch_size, seq_len_k) -> (batch_size, 1, 1, seq_len_k)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3: # (batch_size, seq_len_q, seq_len_k) -> (batch_size, 1, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(1)
            # Mask should now be (batch_size, 1, seq_len_q, seq_len_k) or (batch_size, 1, 1, seq_len_k)

        x, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 4) Concatenate heads and apply final linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.out_linear(x)
        return x, attn_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): Input tensor (batch_size, src_seq_len, d_model)
            src_mask (torch.Tensor, optional): Mask for source sequence. 
                                             For self-attention, this would prevent attending to padding tokens.
                                             Shape (batch_size, src_seq_len) or (batch_size, 1, src_seq_len) 
                                             or (batch_size, src_seq_len, src_seq_len) for more complex masks.
                                             MultiHeadAttention expects (batch_size, 1, 1, src_seq_len) for padding mask.
        Returns:
            torch.Tensor: Output tensor (batch_size, src_seq_len, d_model)
        """
        # Self-attention sublayer
        src_attn_output, _ = self.self_attn(src, src, src, mask=src_mask)
        src = src + self.dropout1(src_attn_output) # Add
        src = self.norm1(src) # Norm

        # Feed-forward sublayer
        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output) # Add
        src = self.norm2(src) # Norm
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, 
                tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            tgt (torch.Tensor): Target input tensor (batch_size, tgt_seq_len, d_model)
            memory (torch.Tensor): Output from the encoder (batch_size, src_seq_len, d_model)
            tgt_mask (torch.Tensor, optional): Mask for the target sequence (e.g., causal mask).
                                             Shape (batch_size, tgt_seq_len, tgt_seq_len).
                                             MultiHeadAttention expects (batch_size, 1, tgt_seq_len, tgt_seq_len).
            memory_mask (torch.Tensor, optional): Mask for the encoder output (e.g., padding mask for source).
                                                Shape (batch_size, 1, src_seq_len).
                                                MultiHeadAttention expects (batch_size, 1, 1, src_seq_len).
        Returns:
            torch.Tensor: Output tensor (batch_size, tgt_seq_len, d_model)
        """
        # Self-attention sublayer (for target sequence)
        tgt_attn_output, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt_attn_output) # Add
        tgt = self.norm1(tgt) # Norm

        # Cross-attention sublayer (target queries, memory keys/values)
        cross_attn_output, _ = self.cross_attn(tgt, memory, memory, mask=memory_mask)
        tgt = tgt + self.dropout2(cross_attn_output) # Add
        tgt = self.norm2(tgt) # Norm

        # Feed-forward sublayer
        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(ff_output) # Add
        tgt = self.norm3(tgt) # Norm
        return tgt


if __name__ == '__main__':
    # Test the components
    batch_size = 2
    seq_len_q = 10
    seq_len_kv = 12 # key/value sequence length can be different for cross-attention
    d_model = config.HIDDEN_SIZE # 512
    num_heads = config.NUM_ATTENTION_HEADS # 8
    d_ff = config.FEED_FORWARD_SIZE # 2048
    dropout_rate = config.DROPOUT_PROB # 0.2

    print(f"Testing Transformer blocks with d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}")

    # --- Positional Encoding Test ---
    pe = PositionalEncoding(d_model, dropout_rate, max_len=50)
    x_pe = torch.randn(batch_size, seq_len_q, d_model)
    output_pe = pe(x_pe)
    print(f"PositionalEncoding Input: {x_pe.shape}, Output: {output_pe.shape}")
    assert output_pe.shape == x_pe.shape

    # --- MultiHeadAttention Test ---
    mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
    query = torch.randn(batch_size, seq_len_q, d_model) # (batch, seq_len_q, d_model)
    key = torch.randn(batch_size, seq_len_kv, d_model)   # (batch, seq_len_k, d_model)
    value = key # (batch, seq_len_v, d_model)
    
    # Test with no mask
    output_mha_nomask, attn_weights_nomask = mha(query, key, value, mask=None)
    print(f"MultiHeadAttention (no mask) Input Q: {query.shape}, K: {key.shape}, V: {value.shape}")
    print(f"MultiHeadAttention (no mask) Output: {output_mha_nomask.shape}, AttnWeights: {attn_weights_nomask.shape}")
    assert output_mha_nomask.shape == (batch_size, seq_len_q, d_model)
    assert attn_weights_nomask.shape == (batch_size, num_heads, seq_len_q, seq_len_kv)

    # Test with padding mask (masking last two tokens of key/value)
    padding_mask = torch.ones(batch_size, seq_len_kv, dtype=torch.bool) # True for non-masked, False for masked
    padding_mask[:, -2:] = False # Mask out last two tokens in key/value sequence
    # Reshape for MHA: (batch_size, 1, 1, seq_len_k)
    # mha_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2) # This is done inside mha now
    output_mha_mask, attn_weights_mask = mha(query, key, value, mask=padding_mask) 
    print(f"MultiHeadAttention (padding mask) Output: {output_mha_mask.shape}, AttnWeights: {attn_weights_mask.shape}")
    assert output_mha_mask.shape == (batch_size, seq_len_q, d_model)
    # Check that attention weights for masked positions are close to zero
    # Sum weights for the masked keys (last 2). For unmasked queries, this sum should be small.
    # This checks that the softmax output for these positions is small.
    masked_key_attn_sum = attn_weights_mask[..., :, -2:].sum(dim=-1) # Sum over the two masked key positions
    assert torch.allclose(masked_key_attn_sum, torch.zeros_like(masked_key_attn_sum), atol=1e-5), \
        f"Attention weights for padded keys are not close to zero. Sum of weights for last 2 keys: {masked_key_attn_sum}"


    # --- PositionwiseFeedForward Test ---
    pff = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
    x_pff = torch.randn(batch_size, seq_len_q, d_model)
    output_pff = pff(x_pff)
    print(f"PositionwiseFeedForward Input: {x_pff.shape}, Output: {output_pff.shape}")
    assert output_pff.shape == x_pff.shape

    # --- TransformerEncoderLayer Test ---
    encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout_rate)
    src_enc = torch.randn(batch_size, seq_len_kv, d_model)
    # Create a source padding mask (mask last token)
    src_enc_mask = torch.ones(batch_size, seq_len_kv, dtype=torch.bool)
    src_enc_mask[:, -1] = False 
    # MHA expects mask as (B, 1, 1, S_k) for padding or (B, 1, S_q, S_k) for causal
    # src_enc_mask_mha = src_enc_mask.unsqueeze(1).unsqueeze(2) # Now handled inside MHA
    output_enc = encoder_layer(src_enc, src_mask=src_enc_mask)
    print(f"TransformerEncoderLayer Input: {src_enc.shape}, Output: {output_enc.shape}")
    assert output_enc.shape == src_enc.shape

    # --- TransformerDecoderLayer Test ---
    decoder_layer = TransformerDecoderLayer(d_model, num_heads, d_ff, dropout_rate)
    tgt_dec = torch.randn(batch_size, seq_len_q, d_model) # Target sequence
    memory_dec = output_enc # Output from encoder layer

    # Create target mask (causal mask for self-attention)
    # (tgt_seq_len, tgt_seq_len)
    tgt_seq_len_dec = tgt_dec.size(1)
    causal_mask = torch.tril(torch.ones(tgt_seq_len_dec, tgt_seq_len_dec, dtype=torch.bool))
    # Broadcast to (batch_size, 1, tgt_seq_len, tgt_seq_len) for MHA
    # causal_mask_mha = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1) # Now handled inside MHA

    # Create memory mask (padding mask from encoder)
    # memory_mask_mha = src_enc_mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, src_seq_len) # Handled inside MHA

    output_dec = decoder_layer(tgt_dec, memory_dec, tgt_mask=causal_mask, memory_mask=src_enc_mask)
    print(f"TransformerDecoderLayer Input Tgt: {tgt_dec.shape}, Memory: {memory_dec.shape}, Output: {output_dec.shape}")
    assert output_dec.shape == tgt_dec.shape

    print("Transformer blocks tested successfully.") 