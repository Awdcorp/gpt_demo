import torch
import torch.nn as nn
from model.attention import MultiHeadSelfAttention

class FeedForward(nn.Module):
    """
    Two-layer feedforward network with GELU activation.
    """
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    A single GPT-style Transformer block.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

    def forward(self, x, return_attn=False, return_trace=False):
        """
        return_attn: return attention weights (for heatmaps)
        return_trace: return intermediate outputs (attn_out, ffn_out)
        """
        # Attention block
        residual = x
        attn_out, attn_weights = self.attn(self.ln1(x), return_attn=True)
        x = residual + attn_out
        attn_output = x  # Store for trace

        # FFN block
        residual = x
        ff_out = self.ff(self.ln2(x))
        x = residual + ff_out
        ffn_output = x  # Store for trace

        if return_trace:
            trace_outputs = {
                "attn_out": attn_output.detach().cpu(),  # shape: (B, T, C)
                "ffn_out": ffn_output.detach().cpu()
            }
            return x, attn_weights, trace_outputs

        if return_attn:
            return x, attn_weights

        return x
