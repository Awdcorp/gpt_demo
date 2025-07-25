import torch
import torch.nn as nn
from model.embedding import GPTEmbedding
from model.block import TransformerBlock

class GPTBackbone(nn.Module):
    """
    Full GPT-style model backbone: embedding → N transformer blocks → LayerNorm.
    Does not include the final language modeling (LM) head yet.
    """
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()

        # Embedding layer (token + position)
        self.embedding = GPTEmbedding(vocab_size, embed_dim, max_seq_len)

        # Stacked Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final LayerNorm (pre-output)
        self.ln_f = nn.LayerNorm(embed_dim)

    def forward(self, input_ids, return_attn=False, return_trace=False):
        """
        input_ids: Tensor of shape (batch_size, seq_len)
        return_attn: If True, return attention weights from each block
        return_trace: If True, return intermediate outputs from each block
        Returns:
            - output embeddings: (B, T, C)
            - attention_weights (optional): list of (B, H, T, T)
            - trace_outputs (optional): list of dicts {'attn_out': ..., 'ffn_out': ...}
        """
        x = self.embedding(input_ids)  # (B, T, C)
        attn_outputs = []
        trace_outputs = []

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

        if return_trace:
            return x, trace_outputs
        if return_attn:
            return x, attn_outputs
        return x

class MiniGPT(nn.Module):
    """
    Full GPT model: embedding → blocks → final LayerNorm → LM head
    Supports optional loss computation if labels are provided.
    """
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.backbone = GPTBackbone(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers, dropout)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)  # Output logits

    def forward(self, input_ids, labels=None, return_attn=False, return_trace=False, return_vectors=False):
        """
        return_vectors: Custom flag used by visualizers to get {'embedding', 'after_attn', 'after_ffn'}
        """

        # === Main forward pass with trace ===
        if return_vectors:
            # Force trace to be captured
            x, trace_outputs = self.backbone(input_ids, return_attn=return_attn, return_trace=True)
            attn_weights = None

            # Collect and return vectors in visualizer-friendly format
            token_vectors = {
                "embedding": self.backbone.embedding(input_ids),  # (B, T, D)
                "after_attn": [layer["attn_out"] for layer in trace_outputs],  # List of (B, T, D)
                "after_ffn": [layer["ffn_out"] for layer in trace_outputs],    # List of (B, T, D)
            }

            logits = self.lm_head(x)

            if labels is not None:
                B, T, V = logits.shape
                loss = nn.functional.cross_entropy(logits.view(B * T, V), labels.view(B * T))
                return logits, loss, attn_weights, token_vectors

            return logits, token_vectors, attn_weights  # ✅ What your visualizer expects

        # === Other existing behavior ===
        if return_trace:
            x, trace_outputs = self.backbone(input_ids, return_attn=return_attn, return_trace=True)
            attn_weights = None
        elif return_attn:
            x, attn_weights = self.backbone(input_ids, return_attn=True)
            trace_outputs = None
        else:
            x = self.backbone(input_ids)
            attn_weights = None
            trace_outputs = None

        logits = self.lm_head(x)

        if labels is not None:
            B, T, V = logits.shape
            loss = nn.functional.cross_entropy(logits.view(B * T, V), labels.view(B * T))
            return logits, loss, attn_weights, trace_outputs

        return logits, None, attn_weights, trace_outputs

