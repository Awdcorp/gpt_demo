import torch
import torch.nn as nn
from model.attention import MultiHeadSelfAttention

class FeedForward(nn.Module):
    """
    Two-layer feedforward network with GELU activation and dropout.
    This is applied after the self-attention block in each Transformer layer.
    """
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),  # Expand dimensionality
            nn.GELU(),                     # Use GELU non-linearity (as in GPT)
            nn.Linear(ff_dim, embed_dim),  # Project back to embedding size
            nn.Dropout(dropout)            # Apply dropout for regularization
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    A single GPT-style Transformer block consisting of:
    - LayerNorm → Self-Attention → Residual Add
    - LayerNorm → FeedForward → Residual Add
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        # === Attention Block ===
        self.ln1 = nn.LayerNorm(embed_dim)                          # Normalize input before attention
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        # === Feedforward Block ===
        self.ln2 = nn.LayerNorm(embed_dim)                          # Normalize before FFN
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

    def forward(self, x, return_attn=False, return_trace=False):
        """
        Args:
            x (Tensor): Input tensor of shape (B, T, C)
            return_attn (bool): If True, also return attention weights
            return_trace (bool): If True, return intermediate outputs for debugging/visualization

        Returns:
            x (Tensor): Output after attention + FFN
            attn_weights (optional): Attention weight matrix (B, num_heads, T, T)
            trace_outputs (optional): Dictionary of intermediate outputs
        """
        # === Self-Attention Block ===
        residual = x
        attn_out, attn_weights = self.attn(self.ln1(x), return_attn=True)  # Pre-norm before attention
        x = residual + attn_out                                            # Residual connection
        attn_output = x                                                    # Save for trace/debug

        # === Feedforward Block ===
        residual = x
        ff_out = self.ff(self.ln2(x))                                      # Pre-norm before FFN
        x = residual + ff_out                                              # Residual connection
        ffn_output = x                                                     # Save for trace/debug

        # === Optional debug/visualization outputs ===
        if return_trace:
            trace_outputs = {
                "attn_out": attn_output.detach().cpu(),  # Freeze and move to CPU for analysis
                "ffn_out": ffn_output.detach().cpu()
            }
            return x, attn_weights, trace_outputs

        if return_attn:
            return x, attn_weights

        return x
