# gpt_model/model/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    """
    Implements masked multi-head self-attention as used in GPT models.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Each head has equal share

        # Linear projections for Q, K, V combined for efficiency
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)

        # Output projection after attention
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=True):
        """
        x: (batch_size, seq_len, embed_dim)
        returns: (batch_size, seq_len, embed_dim)
        """
        B, T, C = x.shape

        # Project input to Q, K, V (batch, seq, 3 * embed_dim)
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (batch, heads, seq, head_dim)

        # Compute scaled dot-product attention scores
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, heads, T, T)

        if mask:
            # Create lower-triangular causal mask (only attend to previous positions)
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # Softmax over the last dim (tokens to attend to)
        attn = F.softmax(scores, dim=-1)
        self.attn_weights = attn.detach().cpu()  # ✅ Store for visualization
        attn = self.attn_dropout(attn)

        # Apply attention weights to values
        out = attn @ v  # (B, heads, T, head_dim)

        # Recombine heads
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)

        # Final output projection
        return self.proj_dropout(self.out_proj(out))
