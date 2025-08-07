# gpt_model/model/embedding.py

import torch
import torch.nn as nn

class GPTEmbedding(nn.Module):
    """
    Embedding layer for GPT-style models.
    Combines:
    - Token embeddings: convert token IDs to vector representations.
    - Positional embeddings: add information about token positions.
    """
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()

        # Learnable lookup table for token embeddings.
        # Each token ID maps to a vector of size embed_dim.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Learnable positional embeddings for each position in the sequence.
        # Enables the model to differentiate token order (since attention is order-agnostic).
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Store the maximum allowed sequence length (used internally or during debugging)
        self.max_seq_len = max_seq_len

    def forward(self, token_ids):
        """
        Forward pass to compute combined embeddings.

        Args:
            token_ids: Tensor of shape (batch_size, seq_len)
                       Contains token indices for each position in the input sequence.

        Returns:
            embeddings: Tensor of shape (batch_size, seq_len, embed_dim)
                        Contains combined token + position embeddings for each input token.
        """
        batch_size, seq_len = token_ids.shape  # Extract batch and sequence length

        # === Token Embeddings ===
        # Look up embeddings for each token ID
        # Shape: (batch_size, seq_len, embed_dim)
        token_embeds = self.token_embedding(token_ids)

        # === Position Embeddings ===
        # Generate position indices for the sequence: [0, 1, 2, ..., seq_len - 1]
        # Then expand to match batch size
        # Shape: (batch_size, seq_len)
        positions = torch.arange(seq_len, device=token_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        # Look up embeddings for each position
        # Shape: (batch_size, seq_len, embed_dim)
        pos_embeds = self.position_embedding(positions)

        # === Combine token + position embeddings ===
        # Element-wise addition encodes both semantic and position info
        return token_embeds + pos_embeds
