# gpt_model/model/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    """
    Implements masked multi-head self-attention as used in GPT models.

    Each token attends to all previous tokens (including itself) using a learned attention mechanism.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Dimension per head

        # Linear layer to compute all Q, K, V in one pass for speed: (B, T, 3C)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)

        # Output projection after multi-head attention
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout after attention and projection
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=True, return_attn=False):
        """
        Args:
            x (Tensor): Input tensor of shape (B, T, C)
            mask (bool): Whether to apply causal mask (for GPT-style decoding)
            return_attn (bool): Whether to return attention weights for analysis

        Returns:
            out (Tensor): Output after applying attention, shape (B, T, C)
            attn (Tensor): Optional attention weights, shape (B, num_heads, T, T)
        """
        B, T, C = x.shape  # Batch, Time, Channels (embed_dim)

        # === Step 1: Project input to Q, K, V (all heads at once) ===
        qkv = self.qkv_proj(x)  # Shape: (B, T, 3 * C)
        
        # Reshape to (3, B, num_heads, T, head_dim) and split
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, T, head_dim)

        # === Step 2: Compute scaled dot-product attention ===
        # scores: (B, num_heads, T, T)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask:
            # Causal mask ensures position t cannot attend to positions > t
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # === Step 3: Softmax over attention scores ===
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)  # Dropout for regularization

        # === Step 4: Apply attention to V ===
        # out: (B, num_heads, T, head_dim)
        out = attn @ v

        # Recombine heads → (B, T, C)
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)

        # Final output projection and dropout
        out = self.proj_dropout(self.out_proj(out))

        # === Return output and optionally attention weights ===
        return (out, attn) if return_attn else (out, None)
