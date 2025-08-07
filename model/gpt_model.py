import torch
import torch.nn as nn
from model.embedding import GPTEmbedding
from model.block import TransformerBlock

class GPTBackbone(nn.Module):
    """
    Full GPT-style model backbone:
    - Embedding layer
    - N stacked Transformer blocks
    - Final LayerNorm

    This excludes the final LM (language modeling) head.
    """
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()

        # === Embedding layer: token + position ===
        self.embedding = GPTEmbedding(vocab_size, embed_dim, max_seq_len)

        # === Stack of Transformer blocks ===
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # === Final layer norm before output ===
        self.ln_f = nn.LayerNorm(embed_dim)

    def forward(self, input_ids, return_attn=False, return_trace=False):
        """
        Args:
            input_ids: Tensor (B, T)
            return_attn: if True, returns attention weights from each block
            return_trace: if True, returns intermediate vectors (for explainability)

        Returns:
            x: final hidden states (B, T, C)
            attn_outputs: list of attention weights (B, H, T, T) per layer (optional)
            trace_outputs: list of {'attn_out', 'ffn_out'} dicts per layer (optional)
        """
        x = self.embedding(input_ids)  # (B, T, C)
        attn_outputs = []
        trace_outputs = []

        # === Pass through transformer blocks ===
        for block in self.blocks:
            if return_trace:
                x, attn, trace = block(x, return_attn=True, return_trace=True)
                trace_outputs.append(trace)
                if return_attn:
                    attn_outputs.append(attn)
            elif return_attn:
                x, attn = block(x, return_attn=True)
                attn_outputs.append(attn)
            else:
                x = block(x)

        x = self.ln_f(x)

        # === Return outputs based on requested flags ===
        if return_trace and return_attn:
            return x, attn_outputs, trace_outputs
        elif return_trace:
            return x, None, trace_outputs
        elif return_attn:
            return x, attn_outputs, None
        else:
            return x, None, None

class MiniGPT(nn.Module):
    """
    Full GPT model:
    - Embedding
    - Transformer blocks
    - Final layer norm
    - LM head for logits

    Supports:
    - Returning attention/trace outputs
    - Returning per-layer vectors for visualizations
    - Loss computation if labels are given
    """
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.backbone = GPTBackbone(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers, dropout)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)  # Final output logits

    def forward(self, input_ids, labels=None, return_attn=False, return_trace=False, return_vectors=False):
        """
        Args:
            input_ids: (B, T) input token IDs
            labels: (B, T) target token IDs for loss computation
            return_attn: return list of attention weights from each layer
            return_trace: return intermediate layer outputs (attn/ffn)
            return_vectors: return dict of {'embedding', 'after_attn', 'after_ffn'}

        Returns:
            logits: (B, T, V) raw token predictions
            loss: optional, cross-entropy loss
            attn_weights: optional list of attention matrices
            trace_outputs: optional list of intermediate vectors or full token_vectors dict
        """
        # === Special visualization case: extract all internal vectors ===
        if return_vectors:
            x, attn_weights, trace_outputs = self.backbone(input_ids, return_attn=True, return_trace=True)

            # Build token vector trace for all stages
            token_vectors = {
                "embedding": self.backbone.embedding(input_ids),                     # (B, T, D)
                "after_attn": [layer["attn_out"] for layer in trace_outputs],        # List of (B, T, D)
                "after_ffn": [layer["ffn_out"] for layer in trace_outputs],          # List of (B, T, D)
            }

            logits = self.lm_head(x)

            if labels is not None:
                B, T, V = logits.shape
                loss = nn.functional.cross_entropy(logits.view(B * T, V), labels.view(B * T))
                return logits, loss, attn_weights, token_vectors

            return logits, token_vectors, attn_weights

        # === Regular forward with optional debug outputs ===
        if return_trace or return_attn:
            x, attn_weights, trace_outputs = self.backbone(input_ids, return_attn=return_attn, return_trace=return_trace)
        else:
            x, attn_weights, trace_outputs = self.backbone(input_ids)

        logits = self.lm_head(x)

        if labels is not None:
            B, T, V = logits.shape
            loss = nn.functional.cross_entropy(logits.view(B * T, V), labels.view(B * T))
            return logits, loss, attn_weights, trace_outputs

        return logits, None, attn_weights, trace_outputs
