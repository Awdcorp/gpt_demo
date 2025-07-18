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

    def forward(self, input_ids):
        """
        input_ids: Tensor of shape (batch_size, seq_len)
        returns: Tensor of shape (batch_size, seq_len, embed_dim)
        """
        x = self.embedding(input_ids)  # (B, T, C)

        for block in self.blocks:
            x = block(x)

        return self.ln_f(x)  # (B, T, C)

class MiniGPT(nn.Module):
    """
    Full GPT model: embedding → blocks → final LayerNorm → LM head
    Supports optional loss computation if labels are provided.
    """
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.backbone = GPTBackbone(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers, dropout)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)  # Output logits

    def forward(self, input_ids, labels=None):
        """
        input_ids: (B, T)
        labels (optional): (B, T) — ground-truth token IDs

        Returns:
            - logits: (B, T, vocab_size)
            - loss (if labels provided)
        """
        x = self.backbone(input_ids)               # (B, T, C)
        logits = self.lm_head(x)                   # (B, T, vocab_size)

        if labels is not None:
            # Flatten inputs for loss: (B*T, vocab)
            B, T, V = logits.shape
            loss = nn.functional.cross_entropy(logits.view(B*T, V), labels.view(B*T))
            return logits, loss

        return logits, None
