# gpt_model/model/gpt_model.py

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

    def forward(self, input_ids, return_attn=False):
        """
        input_ids: Tensor of shape (batch_size, seq_len)
        return_attn: If True, return attention weights from each block
        Returns:
            - output embeddings: (B, T, C)
            - attention_weights (optional): list of (B, H, T, T)
        """
        x = self.embedding(input_ids)  # (B, T, C)
        attn_outputs = []

        for block in self.blocks:
            x, attn = block(x, return_attn=return_attn)
            if return_attn:
                attn_outputs.append(attn)

        x = self.ln_f(x)

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

    def forward(self, input_ids, labels=None, return_attn=False):
        """
        input_ids: (B, T)
        labels (optional): (B, T) — ground-truth token IDs
        return_attn (optional): If True, also return attention weights

        Returns:
            - logits: (B, T, vocab_size)
            - loss (if labels provided)
            - attention_weights (if return_attn=True)
        """
        if return_attn:
            x, attn_weights = self.backbone(input_ids, return_attn=True)
        else:
            x = self.backbone(input_ids)
            attn_weights = None

        logits = self.lm_head(x)  # (B, T, vocab_size)

        if labels is not None:
            B, T, V = logits.shape
            loss = nn.functional.cross_entropy(logits.view(B*T, V), labels.view(B*T))
            return logits, loss, attn_weights

        return logits, None, attn_weights
